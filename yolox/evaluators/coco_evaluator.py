#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
from pycocotools import mask as maskUtils

import numpy as np

import torch

from yolox.data.datasets import BBOX_CLASSES, KEYPOINT_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from typing import List, Dict, Tuple


def per_class_AR_table(coco_eval, class_names=BBOX_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=BBOX_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def get_gt_anns_by_class(cocoGt, class_name: str) -> List[Dict]:
    """按类别获取标注数据"""
    cat_id = [cat['id'] for cat in cocoGt.cats.values() if cat['name'] == class_name][0]
    ann_ids = cocoGt.getAnnIds(catIds=[cat_id])
    return cocoGt.loadAnns(ann_ids)

def get_dt_anns_by_class(cocoDt, class_name: str, cocoGt) -> List[Dict]:
    """按类别获取预测数据"""
    cat_id = [cat['id'] for cat in cocoGt.cats.values() if cat['name'] == class_name][0]
    ann_ids = [ann['id'] for ann in cocoDt.anns.values() if ann['category_id'] == cat_id]
    return [cocoDt.anns[aid] for aid in ann_ids]

def compute_oks(gt_kp: np.ndarray, pred_kp: np.ndarray, gt_area: float, sigma: float) -> float:
    """
    计算单个关键点的OKS（Object Keypoint Similarity）
    Args:
        gt_kp: 标注关键点坐标 [x, y, v]（v=可见性：0不可见，1可见，2标注但不可见）
        pred_kp: 预测关键点坐标 [x, y, score]
        gt_area: 标注框的面积
        sigma: 该关键点的OKS方差
    Returns:
        oks: 单个关键点的OKS值（0~1）
    """
    # 过滤不可见的标注关键点（v=0）
    if gt_kp[2] == 0:
        return 0.0
    
    # 计算欧氏距离
    dx = pred_kp[0] - gt_kp[0]
    dy = pred_kp[1] - gt_kp[1]
    dist_sq = dx * dx + dy * dy
    
    # OKS公式：exp(-dist_sq / (2 * sigma^2 * gt_area))
    oks = np.exp(-dist_sq / (2 * (sigma **2) * gt_area + 1e-9))
    return oks

# 关键点OKS方差配置（与你的setKpParams对应，按KEYPOINT_CLASSES的顺序）
KP_OKS_SIGMAS = {
    "Person": [0.25, 0.25, 0.35, 0.35],  # head, neck, left_shoulder, right_shoulder
    "Seatbelt": [0.25, 0.25],  # top, tail（可根据实际需求调整）
}

def per_keypoint_AP_table(cocoGt, cocoDt, keypoint_class_map=KEYPOINT_CLASSES, 
                          oks_sigmas=KP_OKS_SIGMAS, headers=["class", "keypoint", "KP_AP"], colums=6):
    """
    计算每个类别下单个关键点的AP并生成表格
    Args:
        cocoGt: COCO标注对象
        cocoDt: COCO预测对象
        keypoint_class_map: 关键点类别配置
        oks_sigmas: 每个关键点的OKS方差
        headers: 表格表头
        colums: 表格列数
    Returns:
        格式化的表格字符串
    """
    per_kp_AP = {}
    
    # 遍历所有有关键点的类别
    for class_name, kp_names in keypoint_class_map.items():
        # 获取该类别的标注和预测
        gt_anns = get_gt_anns_by_class(cocoGt, class_name)
        dt_anns = get_dt_anns_by_class(cocoDt, class_name, cocoGt)
        if not gt_anns or not dt_anns:
            for kp_name in kp_names:
                per_kp_AP[(class_name, kp_name)] = float("nan")
            continue
        
        # 获取该类别的OKS方差
        sigmas = oks_sigmas.get(class_name, [0.3]*len(kp_names))
        
        # 遍历该类别下的每个关键点
        for kp_idx, kp_name in enumerate(kp_names):
            sigma = sigmas[kp_idx]
            oks_thresholds = np.linspace(0.5, 0.95, 10)  # COCO标准OKS阈值
            
            # 存储每个OKS阈值下的precision
            all_precisions = []
            
            for oks_thr in oks_thresholds:
                # 统计TP/FP/FN
                tp = 0  # 真阳性
                fp = 0  # 假阳性
                fn = 0  # 假阴性
                
                # 标记已匹配的标注
                gt_matched = [False] * len(gt_anns)
                
                # 遍历预测结果（按score降序）
                dt_anns_sorted = sorted(dt_anns, key=lambda x: x['score'], reverse=True)
                
                for dt_ann in dt_anns_sorted:
                    dt_kp = dt_ann['keypoints'][kp_idx*3 : (kp_idx+1)*3]  # 单个关键点 [x,y,score]
                    if dt_kp[2] < 0.1:  # 预测置信度过低，跳过
                        fp +=1
                        continue
                    
                    # 寻找最佳匹配的标注
                    best_oks = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_ann in enumerate(gt_anns):
                        if gt_matched[gt_idx]:
                            continue
                        gt_kp = gt_ann['keypoints'][kp_idx*3 : (kp_idx+1)*3]  # [x,y,v]
                        gt_area = gt_ann['area'] if 'area' in gt_ann else maskUtils.area(gt_ann['segmentation'])
                        
                        # 计算OKS
                        oks = compute_oks(gt_kp, dt_kp, gt_area, sigma)
                        if oks > best_oks and oks >= oks_thr:
                            best_oks = oks
                            best_gt_idx = gt_idx
                    
                    if best_gt_idx >= 0:
                        tp +=1
                        gt_matched[best_gt_idx] = True
                    else:
                        fp +=1
                
                # 计算FN（未匹配的标注）
                fn = sum([not matched for matched in gt_matched])
                
                # 计算precision
                if tp + fp == 0:
                    precision = 0.0
                else:
                    precision = tp / (tp + fp)
                all_precisions.append(precision)
            
            # 计算该关键点的AP（所有OKS阈值的平均precision）
            all_precisions = np.array(all_precisions)
            ap = np.mean(all_precisions) if all_precisions.size else float("nan")
            per_kp_AP[(class_name, kp_name)] = float(ap * 100)
    
    # 格式化表格数据
    result_list = []
    for (class_name, kp_name), ap_value in per_kp_AP.items():
        result_list.append(class_name)
        result_list.append(kp_name)
        result_list.append(ap_value)
    
    # 生成表格
    num_cols = min(colums, len(result_list))
    row_pair = itertools.zip_longest(*[result_list[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
        missingval="N/A"
    )
    return table

def per_keypoint_AR_table(cocoGt, cocoDt, keypoint_class_map=KEYPOINT_CLASSES, 
                          oks_sigmas=KP_OKS_SIGMAS, headers=["class", "keypoint", "KP_AR"], colums=6):
    """
    计算每个类别下单个关键点的AR并生成表格
    逻辑与AP类似，AR是recall的平均值（recall = TP/(TP+FN)）
    """
    per_kp_AR = {}
    
    # 遍历所有有关键点的类别
    for class_name, kp_names in keypoint_class_map.items():
        gt_anns = get_gt_anns_by_class(cocoGt, class_name)
        dt_anns = get_dt_anns_by_class(cocoDt, class_name, cocoGt)
        if not gt_anns:
            for kp_name in kp_names:
                per_kp_AR[(class_name, kp_name)] = float("nan")
            continue
        
        sigmas = oks_sigmas.get(class_name, [0.3]*len(kp_names))
        
        # 遍历该类别下的每个关键点
        for kp_idx, kp_name in enumerate(kp_names):
            sigma = sigmas[kp_idx]
            oks_thresholds = np.linspace(0.5, 0.95, 10)
            
            all_recalls = []
            for oks_thr in oks_thresholds:
                tp = 0
                fn = 0
                gt_matched = [False] * len(gt_anns)
                
                # 遍历预测结果（取前100个，COCO标准）
                dt_anns_sorted = sorted(dt_anns, key=lambda x: x['score'], reverse=True)[:100]
                
                for dt_ann in dt_anns_sorted:
                    dt_kp = dt_ann['keypoints'][kp_idx*3 : (kp_idx+1)*3]
                    if dt_kp[2] < 0.1:
                        continue
                    
                    best_oks = 0.0
                    best_gt_idx = -1
                    for gt_idx, gt_ann in enumerate(gt_anns):
                        if gt_matched[gt_idx]:
                            continue
                        gt_kp = gt_ann['keypoints'][kp_idx*3 : (kp_idx+1)*3]
                        gt_area = gt_ann['area'] if 'area' in gt_ann else maskUtils.area(gt_ann['segmentation'])
                        
                        oks = compute_oks(gt_kp, dt_kp, gt_area, sigma)
                        if oks > best_oks and oks >= oks_thr:
                            best_oks = oks
                            best_gt_idx = gt_idx
                    
                    if best_gt_idx >= 0:
                        tp +=1
                        gt_matched[best_gt_idx] = True
                
                fn = sum([not matched for matched in gt_matched])
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                all_recalls.append(recall)
            
            all_recalls = np.array(all_recalls)
            ar = np.mean(all_recalls) if all_recalls.size else float("nan")
            per_kp_AR[(class_name, kp_name)] = float(ar * 100)
    
    # 格式化表格
    result_list = []
    for (class_name, kp_name), ar_value in per_kp_AR.items():
        result_list.append(class_name)
        result_list.append(kp_name)
        result_list.append(ar_value)
    
    num_cols = min(colums, len(result_list))
    row_pair = itertools.zip_longest(*[result_list[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
        missingval="N/A"
    )
    return table


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
        keypoints: int = 4,
        per_keypoint_tables: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
            keypoints: 关键点数量。0 表示纯检测模式：后处理/转换/评估
                全部跳过关键点分支（模型输出也不含关键点列）。
            per_keypoint_tables: 是否计算 per-(class, keypoint) AP/AR 表。
                纯 Python 实现，对整个验证集做 OKS 匹配，非常慢，训练时建议关闭。
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.keypoints = keypoints
        self.per_keypoint_tables = per_keypoint_tables

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        if not model.head.decode_in_inference:  ############ 推理后解码
            decoder = model.head.decode_outputs

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs, _ = decoder(outputs, None, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                ###### 后处理
                img_h = info_imgs[0][0]
                img_w = info_imgs[1][0]
                scale = min(
                    self.img_size[0] / float(img_h), 
                    self.img_size[1] / float(img_w)
                )
                h_resized = int(img_h * scale)
                w_resized = int(img_w * scale)
                # 计算居中填充的偏移量（dx：左右偏移，dy：上下偏移）
                # 宽度方向：input_size[1]（640）=w_resized（640），所以dx=0（无需左右填充）
                dx = (self.img_size[1] - w_resized) // 2  
                # 高度方向：input_size[0]（384）-h_resized（360）=24，上下各填12，所以dy=12
                dy = (self.img_size[0] - h_resized) // 2  
                letterbox_info = {
                    "img_h": img_h,
                    "img_w": img_w,
                    "r": scale,
                    "dx": dx,
                    "dy": dy,
                    "input_h": self.img_size[0],
                    "input_w": self.img_size[1],
                }
                outputs = postprocess(
                    outputs, letterbox_info, self.num_classes, self.confthre, self.nmsthre,
                    class_agnostic=False, keypoints=self.keypoints > 0
                )   # 关键点模式 (N, 19) 5+1+1+12；纯检测模式 (N, 7) 5+1+1
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )

            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            if self.keypoints > 0:
                # postprocess 输出布局: (x1,y1,x2,y2, obj, cls_conf, cls_id, kpts...)
                # 关键点从第 7 列开始，共 3*K 列 (x, y, score)
                keypoints = output[:, 7 : 7 + 3 * self.keypoints]
                raw_kpts = keypoints.cpu().numpy().reshape(-1, self.keypoints, 3)
                # x, y 缩放
                raw_kpts[..., 0] /= scale
                raw_kpts[..., 1] /= scale
                # score → visibility
                # score > 0.6 → v = 1, else v = 0
                # raw_kpts[..., 2] = (raw_kpts[..., 2] > 0.6).astype(np.float32)
                # 再展平成 COCO 需要的一维格式
                keypoints = raw_kpts.reshape(-1, 3 * self.keypoints)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                if self.keypoints > 0:
                    pred_data["keypoints"] = keypoints[ind].tolist()  # 直接使用原始12列关键点（已含x/y/v，符合COCO格式）
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, 0, 0, None

        logger.info("Evaluate in main process...")

        # annType = ["segm", "bbox", "keypoints"]
        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")
            
            # 1. 评估检测框（bbox）
            info += "="*50 + " BBOX Evaluation " + "="*50 + "\n"
            cocoEval_bbox = COCOeval(cocoGt, cocoDt, annType[1])  # bbox
            cocoEval_bbox.evaluate()
            cocoEval_bbox.accumulate()
            redirect_string_bbox = io.StringIO()
            with contextlib.redirect_stdout(redirect_string_bbox):
                cocoEval_bbox.summarize()
            bbox_info = redirect_string_bbox.getvalue()
            info += bbox_info + "\n"

            # 2. 评估关键点（keypoints）
            if self.keypoints > 0:
                info += "="*50 + " Keypoints Evaluation " + "="*50 + "\n"
                cocoEval_kp = COCOeval(cocoGt, cocoDt, annType[2])  # keypoints
                # 定义4个关键点的oks方差
                cocoEval_kp.setKpParams(kpt_oks_sigmas=KP_OKS_SIGMAS["Person"])
                # 关键点评估：自动使用OKS阈值，适配4个关键点的数据集
                cocoEval_kp.evaluate()
                cocoEval_kp.accumulate()
                redirect_string_kp = io.StringIO()
                with contextlib.redirect_stdout(redirect_string_kp):
                    cocoEval_kp.summarize()
                kp_info = redirect_string_kp.getvalue()
                info += kp_info + "\n"
                kp_ap50_95, kp_ap50 = cocoEval_kp.stats[0], cocoEval_kp.stats[1]
            else:
                kp_ap50_95, kp_ap50 = 0.0, 0.0

            # 类别级评估（可选）
            info += "="*50 + " BBOX Evaluation (Per class) " + "="*50 + "\n"
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval_bbox, class_names=cat_names)
                info += "BBOX per class AP:\n" + AP_table + "\n\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval_bbox, class_names=cat_names)
                info += "BBOX per class AR:\n" + AR_table + "\n"

            # 4. 单个关键点的AP/AR（纯Python实现很慢，由 per_keypoint_tables 控制，训练时默认关闭）
            if self.keypoints > 0 and self.per_keypoint_tables:
                info += "="*50 + " Keypoints Evaluation (Per Keypoint) " + "="*50 + "\n"
                kp_AP_table = per_keypoint_AP_table(
                    cocoGt=cocoGt,
                    cocoDt=cocoDt,
                    keypoint_class_map=KEYPOINT_CLASSES,
                    oks_sigmas= KP_OKS_SIGMAS
                )
                info += "Keypoint per (class, keypoint) AP:\n" + kp_AP_table + "\n\n"

                kp_AR_table = per_keypoint_AR_table(
                    cocoGt=cocoGt,
                    cocoDt=cocoDt,
                    keypoint_class_map=KEYPOINT_CLASSES,
                    oks_sigmas= KP_OKS_SIGMAS
                )
                info += "Keypoint per (class, keypoint) AR:\n" + kp_AR_table + "\n"

            # 返回bbox AP、关键点AP、完整日志
            return cocoEval_bbox.stats[0], cocoEval_bbox.stats[1], kp_ap50_95, kp_ap50, info
        else:
            return 0, 0 ,0, 0, info
