#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision
import torch.nn.functional as F

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "adjust_lmks_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "bboxes_iou_batch",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, letterbox_info, num_classes, conf_thre=0.7, nms_thre=0.45,
                class_agnostic=False, keypoints=False, segs=False):
    """
    适配Letterbox预处理的后处理函数
    Args:
        prediction: 模型输出，shape=[B, 5040, 22]，22列=[4(reg)+1(obj)+5(cls)+12(kpts)]
        letterbox_info: 每个图像的Letterbox信息，格式为列表（len=B），每个元素是字典：
            {
                "img_h": 原始图像高度,
                "img_w": 原始图像宽度,
                "r": Letterbox缩放比例,
                "dx": 水平偏移量,
                "dy": 垂直偏移量,
                "input_h": Letterbox目标高度,
                "input_w": Letterbox目标宽度,
            }
        num_classes: 类别数（对应cls_output的5列）
        conf_thre: 置信度阈值
        nms_thre: NMS阈值
        class_agnostic: 是否类别无关NMS
        keypoints: 是否处理关键点
        segs: 是否处理分割（暂未实现）
    Returns:
        output: 后处理后的检测结果，len=B，每个元素shape=[N, 6+12]（xyxy, obj_conf, cls_conf, cls_id, kpts）
                坐标已还原到模型输入图像尺寸
                坐标范围: [360,640]
    """
    # 1. 先将cxcywh转为xyxy（Letterbox尺寸下的坐标）
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # 提取当前图像的Letterbox参数
    img_h = letterbox_info["img_h"]
    img_w = letterbox_info["img_w"]
    r = letterbox_info["r"]
    dx = letterbox_info["dx"]
    dy = letterbox_info["dy"]
    model_input_h = letterbox_info["input_h"]
    model_input_w = letterbox_info["input_w"]
    for i, image_pred in enumerate(prediction):
        # 空检测结果直接跳过
        if not image_pred.size(0):
            continue
        
        # 2. 筛选高置信度结果
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        
        # 拼接检测结果：(x1,y1,x2,y2,obj_conf,cls_conf,cls_id, kpts...)
        if keypoints or segs:
            detections = torch.cat((
                image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5+num_classes:]
            ), 1)
        else:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # 5. NMS非极大值抑制
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
        detections = detections[nms_out_index]
        if not detections.size(0):
            continue

        # ===================== 核心修正：Letterbox坐标反向还原 =====================
        # 4.1 检测框坐标还原（xyxy → 原始图像尺寸）
        # 公式：原始坐标 = (Letterbox坐标 - 偏移量) / 缩放比例
        detections[:, 0] = (detections[:, 0] - dx) #/ r  # x1 还原 为1280*720
        detections[:, 1] = (detections[:, 1] - dy) #/ r  # y1 还原
        detections[:, 2] = (detections[:, 2] - dx) #/ r  # x2 还原
        detections[:, 3] = (detections[:, 3] - dy) #/ r  # y2 还原

        # 4.2 关键点坐标还原（12列：7~19列，每2列对应1个关键点的x/y）
        if keypoints and detections.shape[1] >= 7 + 12:
            kpts = detections[:, 7:19]  # 提取12列关键点（Letterbox尺寸）
            # x列（0/2/4/6/8/10）：减dx后除以r
            kpts[:, 0::3] = (kpts[:, 0::3] - dx) #/ r
            # y列（1/3/5/7/9/11）：减dy后除以r
            kpts[:, 1::3] = (kpts[:, 1::3] - dy) #/ r
            # 对齐检测框的裁剪范围（img_w*r / img_h*r），避免关键点超出图像范围
            kpts[:, 0::3] = torch.clamp(kpts[:, 0::3], 0, img_w * r)  # x坐标 ∈ [0, img_w*r]
            kpts[:, 1::3] = torch.clamp(kpts[:, 1::3], 0, img_h * r)  # y坐标 ∈ [0, img_h*r]
            # 放回detections
            detections[:, 7:19] = kpts

        # 4.3 坐标边界裁剪（避免超出原始图像范围）
        detections[:, 0] = torch.clamp(detections[:, 0], 0, img_w * r)  # x1 ∈ [0, img_w]
        detections[:, 1] = torch.clamp(detections[:, 1], 0, img_h * r)  # y1 ∈ [0, img_h]
        detections[:, 2] = torch.clamp(detections[:, 2], 0, img_w * r)  # x2 ∈ [0, img_w]
        detections[:, 3] = torch.clamp(detections[:, 3], 0, img_h * r)  # y2 ∈ [0, img_h]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    
    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox

def adjust_lmks_anns(landmarks, scale_ratio, padw, padh, w_max, h_max):
    mask_landmarks = np.array(landmarks > 0, dtype=np.int32)
    landmarks[:, 0::2] = landmarks[:, 0::2] * scale_ratio + padw
    landmarks[:, 1::2] = landmarks[:, 1::2] * scale_ratio + padh
    landmarks = landmarks * mask_landmarks + mask_landmarks - 1
    for j, points in enumerate(landmarks):
        points = points.reshape(-1, 2)
        for k, point in enumerate(points):
            x, y = point
            if x < 0 or y < 0 or x > w_max or y > h_max:
                landmarks[j][2 * k] = -1
                landmarks[j][2 * k + 1] = -1
    return landmarks


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    N = bboxes_a.shape[0]
    A = bboxes_a.shape[1]
    B = bboxes_b.shape[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = torch.cat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                           bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], dim=-1)
        box_b = torch.cat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                           bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], dim=-1)

    box_a_rb = torch.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = torch.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = torch.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = torch.tile(box_b_rb, [1, A, 1, 1])
    max_xy = torch.minimum(box_a_rb, box_b_rb)

    box_a_lu = torch.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = torch.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = torch.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = torch.tile(box_b_lu, [1, A, 1, 1])
    min_xy = torch.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = torch.reshape(area_a, (N, A, 1))
    area_a = torch.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = torch.reshape(area_b, (N, 1, B))
    area_b = torch.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]
