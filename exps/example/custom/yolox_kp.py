#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.exp import Exp as MyExp

import numpy as np

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # #### n
        self.depth = 0.33
        self.width = 0.25
        # #### s
        # self.depth = 0.33
        # self.width = 0.50
        #### m
        # self.depth = 0.67
        # self.width = 0.75
        #### l
        # self.depth = 1.0
        # self.width = 1.0
        #### x
        # self.depth = 1.33
        # self.width = 1.25
        # ---------------- model config ---------------- #
        self.num_classes = 5
        self.keypoints = 4
        self.segcls = 0
        # self.depth = 0.33
        # self.width = 0.25
        self.act = 'relu'
        self.img_channel = 3
        self.in_channels = [256, 512, 1024]
        self.in_features = ("dark3", "dark4", "dark5")
        self.model_name = 'yolo_pafpn'
        self.repeat = 2
        self.decode_in_inference = False
        self.data_num_workers = 16
        self.input_size = (384, 640)  # (height, width)
        self.multiscale_range = 5

        scale_h, scale_w = np.array(self.input_size) / 32
        if isinstance(self.multiscale_range, int):
            assert scale_h > self.multiscale_range and scale_w > self.multiscale_range
        else:
            assert scale_h > self.multiscale_range[0] and scale_w > self.multiscale_range[1]

        self.pin_memory = True
        # 自定义随机生成器
        self.random_dataset = None
        self.cate_ls, self.mask_order, self.cate_id = [], [], []

        self.data_dir = 'datasets/YaXon_DMS_OD_Keypoint'
        self.train_ann = "person_keypoints_train2017.json" # person_keypoints_train2017_512
        self.val_ann = "person_keypoints_val2017.json"  # person_keypoints_val2017_256
        self.finetuning_train_ann = "person_keypoints_train2017.json"  # 微调数据集标签
        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.4#
        self.mixup_prob = 0
        self.hsv_prob = 0.7
        self.flip_prob = 0.7
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 1
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 10
        # self.max_epoch = 1300
        self.warmup_lr = 0
        self.adam = False
        self.basic_lr_per_img = 5e-6  # QAT训练  #0.001 / 32.0 if self.adam else 0.01 / 32.0
        self.scheduler = "yoloxwarmcos"
        self.aug_epochs = 1000              # 数据增强训练轮次
        self.no_aug_epochs = 300            # 非数据增强训练轮次
        self.finetuning_train_epoch = 0   # 微调数据集训练轮次
        self.max_epoch = self.aug_epochs + self.no_aug_epochs + self.finetuning_train_epoch     # 总训练轮次

        self.min_lr_ratio = 0.1 #05
        self.ema = True

        self.weight_decay = 5e-5 #4e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.no_aug_eval_interval = 5   # no-aug 阶段的评估间隔（trainer 之前强制为 1）
        self.eval_per_kp = False        # 纯 Python 的 per-keypoint AP/AR 表很慢，训练时默认关闭；-o eval_per_kp True 开启
        self.disable_kp_loss = False    # 训练时不计算关键点 loss：kp 头结构/输出/评估均不变（ONNX/板端不改），kp 参数无梯度 ≈ 冻结
        self.save_history_ckpt = False
        self.save_history_ckpt_interval = 100
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (384, 640)
        self.test_conf = 0.7
        self.nmsthre = 0.30
        self.model_export = False
        self.output_dir = f'./YOLOX_outputs/'


    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, YOLOPAFPNSLIM, YOLO7TINY
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            if self.model_name == 'CoAtNet_pafpn':
                backbone = YOLOPAFPN(self.img_channel, self.depth, self.width, in_channels=self.in_channels,
                                    in_features=self.in_features, backbone_name='CoAtNet', act=self.act,
                                    input_size=self.input_size)
            elif self.model_name == 'yolo_pafpn_slim':
                backbone = YOLOPAFPNSLIM(self.img_channel, self.depth, self.width, in_channels=self.in_channels,
                                        in_features=self.in_features, act=self.act,
                                        input_size=self.input_size)
            elif self.model_name == 'yolov7_tiny':
                self.in_channels = [256, 512, 1024]
                backbone = YOLO7TINY(self.img_channel, self.depth, self.width, in_channels=self.in_channels,
                                    in_features=self.in_features, act=self.act,
                                    input_size=self.input_size)
            else:
                backbone = YOLOPAFPN(self.img_channel, self.depth, self.width, in_channels=self.in_channels,
                                    in_features=self.in_features, act=self.act,
                                    depthwise=False,
                                    input_size=self.input_size)

            head = YOLOXHead(self.num_classes, self.width, in_channels=self.in_channels, act=self.act,
                            keypoints=self.keypoints, segcls=self.segcls, model_export=self.model_export,
                            repeat=self.repeat,
                            depthwise=True,
                            decode_in_inference = self.decode_in_inference)
            # 把 kp loss 开关传给 head（只训检测框、保持 ONNX 结构/板端代码不变时用）
            head.disable_kp_loss = getattr(self, "disable_kp_loss", False)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                keypoints=self.keypoints,
                segcls=self.segcls,
                random_dataset=self.random_dataset,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    keypoints=self.keypoints,
                    segcls=self.segcls),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,  # 是否使用增强
            img_size=self.input_size,
            keypoints=self.keypoints,
            segcls=self.segcls,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                keypoints=self.keypoints,
                segcls=self.segcls),
            degrees=self.degrees,  # 旋转角度 10
            translate=self.translate,  # 平移 0.1
            mosaic_scale=self.mosaic_scale,  # 尺度 (0.1, 2)
            mixup_scale=self.mixup_scale,  # 尺度 (0.5, 1.5)
            shear=self.shear,  # 裁剪 2.0
            enable_mixup=self.enable_mixup,  # 是否使用mixup
            mosaic_prob=self.mosaic_prob,  # 概率 1.0
            mixup_prob=self.mixup_prob,  # 概率 1.0
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    # 用来最后微调的数据集，一般是样本少的巡检误报视频，由参数
    def get_data_loader_finetuning(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.finetuning_train_ann,
                img_size=self.input_size,
                keypoints=self.keypoints,
                segcls=self.segcls,
                random_dataset=self.random_dataset,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    keypoints=self.keypoints,
                    segcls=self.segcls),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,  # 是否使用增强
            img_size=self.input_size,
            keypoints=self.keypoints,
            segcls=self.segcls,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                keypoints=self.keypoints,
                segcls=self.segcls),
            degrees=self.degrees,  # 旋转角度 10
            translate=self.translate,  # 平移 0.1
            mosaic_scale=self.mosaic_scale,  # 尺度 (0.1, 2)
            mixup_scale=self.mixup_scale,  # 尺度 (0.5, 1.5)
            shear=self.shear,  # 裁剪 2.0
            enable_mixup=self.enable_mixup,  # 是否使用mixup
            mosaic_prob=self.mosaic_prob,  # 概率 1.0
            mixup_prob=self.mixup_prob,  # 概率 1.0
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader


    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                if isinstance(self.multiscale_range, int):
                    min_size = int(self.input_size[0] / 32) - self.multiscale_range
                    max_size = int(self.input_size[0] / 32) + self.multiscale_range
                else:
                    min_size = int(self.input_size[0] / 32) - self.multiscale_range[0]
                    max_size = int(self.input_size[0] / 32) + self.multiscale_range[1]
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize, seg_targets):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
            if self.segcls > 0:
                seg_targets = nn.functional.interpolate(seg_targets, size=tsize, mode="nearest")
        return inputs, targets, seg_targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay
            if self.adam:
                optimizer = torch.optim.AdamW(pg0, lr=lr)
            else:
                optimizer = torch.optim.SGD(
                    pg0, lr=lr, momentum=self.momentum, nesterov=True
                )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform, worker_init_reset_seed

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "sampler": sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            per_class_AP = True,
            per_class_AR = True,
            keypoints = self.keypoints,
            per_keypoint_tables = self.eval_per_kp,
        )
        return evaluator

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
