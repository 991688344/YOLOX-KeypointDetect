#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger
from typing import Dict  # 先导入 Dict

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)

import cv2
import numpy as np
import torch
import os
from typing import Union


class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        self.kp_best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        # self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name_with_timestamp = f"{args.experiment_name}_{timestamp}"
        # 拼接完整路径
        self.file_name = os.path.join(exp.output_dir, exp_name_with_timestamp)

        # 检查目录是否存在，不存在则创建（exist_ok=True 避免目录已存在时报错）
        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets, seg_targets = self.prefetcher.next()  # 从dataloader获取数据
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        seg_targets = seg_targets.to(self.data_type)
        targets.requires_grad = False
        seg_targets.requires_grad = False

        inps, targets, seg_targets = self.exp.preprocess(inps, targets, self.input_size, seg_targets)
        data_end_time = time.time()

        # with torch.cuda.amp.autocast(enabled=self.amp_training):
        with torch.amp.autocast('cuda', enabled=self.amp_training):
            outputs, seg_output = self.model(inps, targets, seg_targets)  #

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        try:
            logger.info(
                "Model Summary: {}".format(get_model_info(model, self.exp.test_size, self.exp.img_channel))
            )
        except:
            pass
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.exp.aug_epochs   # self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        self.train_loader_finetuning = self.exp.get_data_loader_finetuning(     # 微调数据加载器
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.prefetcher_finetuning = DataPrefetcher(self.train_loader_finetuning)   # 微调数据预取器
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        self.max_iter_finetuning = len(self.train_loader_finetuning)    # 微调数据迭代次数

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    #### 负责根据epoch数量来切换数据增强和微调设置 ####
    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.exp.aug_epochs or self.no_aug:    #  - self.exp.no_aug_epochs 
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")
        ### 切换到微调 ###
        if self.epoch + 1 == self.exp.aug_epochs + self.exp.no_aug_epochs:    #  - self.exp.no_aug_epochs
            logger.info("--->Start finetuning now!")
            self.train_loader_finetuning.close_mosaic()  # 确保微调数据加载器不使用mosaic
            self.prefetcher = self.prefetcher_finetuning
            self.max_iter = self.max_iter_finetuning

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # # resume the model/optimizer state dict
            # model.load_state_dict(ckpt["model"])
            # self.optimizer.load_state_dict(ckpt["optimizer"])

            # -------------------------- 处理模型参数加载 --------------------------
            model_state_dict = model.state_dict()
            pretrained_model_dict = ckpt["model"]
            # 过滤模型参数（保留名称和形状均匹配的参数）
            filtered_model_dict = {
                k: v for k, v in pretrained_model_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            # 加载过滤后的模型参数
            model.load_state_dict(filtered_model_dict, strict=False)
            logger.info("Loaded model weights (unmatched parameters skipped)")
            # -------------------------- 处理模型参数加载 END--------------------------

            self.best_ap = ckpt.pop("best_ap", 0)
            logger.info(f"Best ap from pretrain is {self.best_ap:.4f}")
            self.best_ap = 0
            logger.info(f"Best ap reset to {self.best_ap:.4f}")
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    # per class AP/AR in tensorboard logs
    def parse_per_class_table(self, table_text: str) -> Dict[str, float]:  # 这里用 Dict 替代 dict
        metrics = {}
        for raw_line in table_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue
            if ":---" in line or "class" in line.lower():
                continue

            parts = [p.strip() for p in line.strip("|").split("|")]
            for i in range(0, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                cls_name, val_str = parts[i], parts[i + 1]
                if not cls_name or not val_str:
                    continue
                try:
                    metrics[cls_name] = float(val_str)
                except ValueError:
                    continue
        return metrics

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, kp_ap50_95, kp_ap50, summary) = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed
            )

        update_best_ckpt = ap50_95 > self.best_ap or kp_ap50_95 > self.kp_best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        self.kp_best_ap = max(self.kp_best_ap, kp_ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
                self.tblogger.add_scalar("val/KPAP50", kp_ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/KPAP50_95", kp_ap50_95, self.epoch + 1)
                try:
                    if "per class AP:" in summary:
                        ap_section = summary.split("per class AP:", 1)[1]
                        ap_table_text = ap_section.split("per class AR:", 1)[0].strip()
                        per_class_ap = self.parse_per_class_table(ap_table_text)
                        for cls_name, ap in per_class_ap.items():
                            self.tblogger.add_scalar(f"per_class_AP_val/{cls_name}", ap, self.epoch + 1)

                    if "per class AR:" in summary:
                        ar_section = summary.split("per class AR:", 1)[1]
                        ar_table_text = ar_section.strip()
                        per_class_ar = self.parse_per_class_table(ar_table_text)
                        for cls_name, ar in per_class_ar.items():
                            self.tblogger.add_scalar(f"per_class_AR_val/{cls_name}", ar, self.epoch + 1)
                except Exception as e:
                    pass
            logger.info("\n" + summary)
        synchronize()

        # if self.rank == 0:
        #     if self.args.logger == "tensorboard":
        #         self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
        #         self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
        #     if self.args.logger == "wandb":
        #         self.wandb_logger.log_metrics({
        #             "val/COCOAP50": ap50,
        #             "val/COCOAP50_95": ap50_95,
        #             "train/epoch": self.epoch + 1,
        #         })
        #         self.wandb_logger.log_images(predictions)
        #     logger.info("\n" + summary)
        # synchronize()
        # update_best_ckpt, ap50_95 = False, None
        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95, kp_ap50_95=kp_ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95, kp_ap50_95=kp_ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None, kp_ap50_95=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "kp_best_ap": self.kp_best_ap,
                "curr_ap": ap,
                "curr_kp_ap": kp_ap50_95
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap,
                        "curr_kp_ap": kp_ap50_95
                    }
                )
        
    def visualize_model_input(self,
                            img_tensor: torch.Tensor, 
                            target_tensor: torch.Tensor, 
                            save_path: str,
                            norm_mean: tuple = (0.0, 0.0, 0.0),  
                            norm_std: tuple = (1.0, 1.0, 1.0),
                            bbox_format: str = "xyxy"):
        """
        （类方法）可视化模型输入的第0张图片和对应标签（修复OpenCV绘图兼容问题）
        Args:
            self: 类实例（自动传入）
            img_tensor: 单张图片张量，shape=[3, H, W]（如[3,384,640]）
            target_tensor: 单张图片标签张量，shape=[120, 13]
            save_path: 保存路径
            norm_mean: 图像归一化均值
            norm_std: 图像归一化标准差
            bbox_format: bbox格式（xyxy/xywh）
        """
        # ===================== 1. 数据预处理：确保兼容OpenCV =====================
        # 1.1 张量转numpy（解耦+CPU）
        img_np = img_tensor.detach().cpu().numpy()  # [3, 384, 640]
        target_np = target_tensor.detach().cpu().numpy()  # [120, 17] 5+12
        
        # 1.2 CHW转HWC + 还原归一化 + 限制范围 + 转uint8
        img_np = img_np.transpose(1, 2, 0)  # [H, W, 3]
        img_np = img_np * np.array(norm_std).reshape(1, 1, 3) + np.array(norm_mean).reshape(1, 1, 3)
        img_np = np.clip(img_np, 0, 255)  # 确保像素值在0-255
        img_np = img_np.astype(np.uint8)   # 转uint8
        
        # 1.3 关键修复：转为连续数组（OpenCV必需）
        img_np = np.ascontiguousarray(img_np)
        
        # 1.4 校验图像格式（调试用，可保留）
        print(f"📌 图像格式校验：shape={img_np.shape}, dtype={img_np.dtype}, contiguous={img_np.flags['C_CONTIGUOUS']}")
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            raise ValueError(f"图像格式错误！需为HWC彩色图，当前shape={img_np.shape}")

        # ===================== 2. 过滤有效标签 =====================
        valid_mask = ~np.all(target_np == 0, axis=1)
        valid_targets = target_np[valid_mask]
        if len(valid_targets) == 0:
            print("⚠️  当前图片无有效标签，仅保存原图")
            cv2.imwrite(save_path, img_np)
            return

        # ===================== 3. 解析标签并绘制（修复参数传递） =====================
        COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        H, W = img_np.shape[:2]

        for idx, target in enumerate(valid_targets):
            # 3.1 解析标签
            cls_id = int(target[0])
            bbox = target[1:5]
            keypoints = target[5:17]

            # 3.2 解析bbox
            if bbox_format == "xyxy":
                x1, y1, x2, y2 = bbox.astype(int)
            elif bbox_format == "xywh":
                cx, cy, w, h = bbox
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
            else:
                raise ValueError("bbox_format仅支持'xyxy'或'xywh'")

            # 3.3 严格坐标校验（避免越界）
            x1 = np.clip(x1, 0, W-1)
            y1 = np.clip(y1, 0, H-1)
            x2 = np.clip(x2, 0, W-1)
            y2 = np.clip(y2, 0, H-1)

            # 3.4 绘制检测框（修复参数传递：去掉关键字参数，用位置参数）
            color = COLORS[cls_id % len(COLORS)]
            # 规范写法：所有参数按位置传递，避免OpenCV解析冲突
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)  # 去掉thickness=2，改为位置参数

            # 3.5 绘制文本（同样规范参数）
            text = f"cls:{cls_id} idx:{idx}"
            cv2.putText(img_np, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3.6 绘制关键点
            kps = keypoints.reshape(-1, 3)
            for kp_idx, (kx, ky, vis) in enumerate(kps):
                if kx <= 0 or ky <= 0 or kx >= W or ky >= H:
                    continue
                kx, ky = int(kx), int(ky)
                cv2.circle(img_np, (kx, ky), 4, color, -1)  # 去掉radius/thickness关键字
                cv2.putText(img_np, str(kp_idx), (kx+5, ky+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ===================== 4. 保存图像 =====================
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)  # 加exist_ok=True避免重复创建报错
        # 保存前再次确保数组连续
        img_np = np.ascontiguousarray(img_np)
        cv2.imwrite(save_path, img_np)
        print(f"✅  可视化结果已保存到：{save_path}")
        print(f"   有效标签数量：{len(valid_targets)}")