#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np
from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def add_gaussian_noise(img, sigma=4.0):
    """高斯加性噪声，作用在 [0,255] float 图像上（HWC/CHW 均可），输出裁剪到 [0,255]。

    与 eye 分类项目参数一致：sigma=4.0 在 [0,255] 域 ≈ 归一化域的 0.016，非常温和。
    本管线 preproc 后图像为 float32 [0,255]（不除以 255），故 sigma 按 [0,255] 域取值。
    """
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img + noise, 0, 255).astype(np.float32)


def add_salt_pepper(img, p=0.004, salt_vs_pepper=0.5):
    """椒盐噪声，作用在 [0,255] float32 CHW 图像上。

    p: 被污染像素比例（每个空间位置一次随机，三通道共享掩码，模拟真实 sensor 坏点）；
    salt_vs_pepper: 污染像素中盐噪(255)的占比，其余为椒噪(0)。
    p=0.004 时 640x384 图约 983 像素/通道被污染，视觉上几乎不可察觉。
    """
    h, w = img.shape[1], img.shape[2]
    mask = np.random.rand(h, w)
    salt = mask < p * salt_vs_pepper
    pepper = (mask >= p * salt_vs_pepper) & (mask < p)
    out = img.copy()
    out[:, salt] = 255.0   # 盐噪置 255
    out[:, pepper] = 0.0   # 椒噪置 0
    return out


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:计算候选框，包括以下5项
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    # 放大前的框1，放大后的框2，像素，纵横比，面积比
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates


def random_perspective(
        img,
        targets=(),
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=10,
        perspective=0.0,
        border=(0, 0),
        keypoints=0,
        kp_dim=3,             # 每个关键点的维度 (x,y,v)
        segcls=0,
        seg=np.array([])
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2  # border=[-input_h // 2, -input_w // 2],

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:  # False
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # this
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )
            if segcls > 0:
                seg = cv2.warpAffine(
                    seg, M[:2], dsize=(int(width), int(height)), borderValue=(0, 0, 0), flags=cv2.INTER_NEAREST
                )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:  # False
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale 除以w
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        ind = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[ind]
        targets[:, :4] = xy[ind]
        m = len(targets)
        # 适配keypoints从4*2 → 4*3 (坐标+可见性)
        if keypoints > 0 and m > 0:
            # 1. 提取keypoints部分 (最后3*keypoints列)
            kpts = targets[:, -3 * keypoints:]  # 原代码是-2*keypoints，改为3倍
            # 2. 拆分坐标(x,y)和可见性(vis)：reshape为(m, keypoints, 3)
            kpts_reshaped = kpts.reshape(m, keypoints, 3)
            kpts_xy = kpts_reshaped[..., :2]  # 坐标部分 (m, keypoints, 2)
            kpts_vis = kpts_reshaped[..., 2:]  # 可见性部分 (m, keypoints, 1) → 不参与任何变换
            
            # 3. 处理坐标变换（仅对x,y）
            landmarks_xy = kpts_xy.reshape(m * keypoints, 2)  # 展平为(m*keypoints, 2)
            landmarks = np.ones((m * keypoints, 3))  # 构造齐次坐标
            landmarks[:, :2] = landmarks_xy
            
            # 4. 掩码处理（仅针对坐标，过滤无效点）
            mask_landmarks = (landmarks[:, :2] > 0).astype(np.int32)  # 向量化处理，替代原列表推导式
            landmarks = landmarks @ M.T  # 应用变换矩阵
            
            # 5. 掩码修正（保持原逻辑）
            landmarks[:, :2] = landmarks[:, :2] * mask_landmarks + mask_landmarks - 1

            if perspective:  # 透视变换
                # 归一化坐标
                landmarks[:, :2] = landmarks[:, :2] / landmarks[:, 2:3]
                # 重新生成掩码
                mask_landmarks = (landmarks[:, :2] != -1).astype(np.int32)
                landmarks[:, :2] = landmarks[:, :2] * mask_landmarks + mask_landmarks - 1
                # 重塑为(m, keypoints, 2)
                transformed_xy = landmarks[:, :2].reshape(m, keypoints, 2)
            else:  # 仿射变换（主流场景）
                # 直接取前两列，重塑为(m, keypoints, 2)
                transformed_xy = landmarks[:, :2].reshape(m, keypoints, 2)

            # 6. 合并变换后的坐标 + 原始可见性（核心：可见性不变）
            transformed_kpts = np.concatenate([transformed_xy, kpts_vis], axis=-1)
            # 7. 重塑并赋值回targets
            targets[:, -3 * keypoints:] = transformed_kpts.reshape(m, 3 * keypoints)
        # if keypoints > 0 and m > 0:
        #     landmarks = np.ones((m * keypoints, 3))
        #     landmarks[:, :2] = targets[:, -2 * keypoints:].reshape(m * keypoints, 2)
        #     mask_landmarks = [np.array(x > 0, dtype=np.int32) for x in landmarks]
        #     landmarks = landmarks @ M.T  # transform
        #     landmarks = np.array([x * y + y - 1 for x, y in zip(landmarks, mask_landmarks)])

        #     if perspective:  # False
        #         mask_landmarks = np.array([np.array(x != -1, dtype=np.int32) for x in landmarks[:, :2]]).reshape(m,
        #                                                                                                          2 * keypoints)
        #         landmarks = (landmarks[:, :2] / landmarks[:, 2:3]).reshape(m, 2 * keypoints)  # rescale
        #         landmarks = np.array([x * y + y - 1 for x, y in zip(landmarks, mask_landmarks)])
        #     else:  # affine
        #         landmarks = landmarks[:, :2].reshape(m, 2 * keypoints)
        #     targets[:, -2 * keypoints:] = landmarks
    return img, targets, seg


def _mirror(image, boxes, landmarks, segs, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob and len(landmarks) == 0:  # 关键点检测不支持翻转
        image = image[:, ::-1]
        segs = segs[:, ::-1] if len(segs) else np.array([])
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # add for 5 landmarks  使用镜像的时候要注意对关键点进行重新排位
        # if len(landmarks) > 0:
        #     landmarks[:, 0::2] = width - landmarks[:, 2::-2]
    return image, boxes, landmarks, segs


def preproc(img, input_size, seg_target, swap=(2, 0, 1)):
    """
    标准的居中Letterbox预处理
    Args:
        img: 输入图像 (HWC格式)
        input_size: 目标尺寸 (height, width)
        seg_target: 分割标签（空列表/数组）
        swap: 维度转换参数，默认(2,0,1)将HWC转为CHW
    Returns:
        padded_img: 居中填充后的图像
        r: 缩放比例
        padded_seg: 居中填充后的分割标签
    """
    h, w, c = img.shape
    # 1. 创建目标尺寸的填充画布（背景值114）
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], c), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    # 2. 计算等比例缩放因子（核心：保持宽高比）
    r = min(input_size[0] / h, input_size[1] / w)
    # 3. 等比例缩放图像
    resized_h, resized_w = int(h * r), int(w * r)
    resized_img = cv2.resize(
        img,
        (resized_w, resized_h),  # cv2.resize的参数是(w, h)，注意顺序
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    
    # 4. 计算居中偏移量（标准Letterbox核心修改点）
    dy = (input_size[0] - resized_h) // 2  # 垂直方向偏移（上下均分空白）
    dx = (input_size[1] - resized_w) // 2  # 水平方向偏移（左右均分空白）
    # 5. 居中放置缩放后的图像
    padded_img[dy:dy+resized_h, dx:dx+resized_w] = resized_img
    
    # 6. 格式转换：HWC -> CHW，转为连续数组并调整数据类型
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    # 7. 分割标签的居中Letterbox处理（与图像逻辑完全一致）
    padded_seg = seg_target
    if len(seg_target) > 0:
        h_seg, w_seg, segcls = seg_target.shape
        padded_seg = np.zeros((int(input_size[0]), int(input_size[1]), segcls), dtype=np.uint8)
        # 等比例缩放分割标签（用最近邻插值避免类别值失真）
        resized_seg = cv2.resize(
            seg_target, (resized_w, resized_h),  # 与图像缩放尺寸一致
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        if len(resized_seg.shape) == 2:
            resized_seg = np.expand_dims(resized_seg, axis=-1)
        # 居中放置分割标签（核心修改点）
        padded_seg[dy:dy+resized_h, dx:dx+resized_w] = resized_seg
        # 格式转换
        padded_seg = padded_seg.transpose(swap)
        padded_seg = np.ascontiguousarray(padded_seg, dtype=np.float32)

    return padded_img, padded_seg, r, dx, dy


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, keypoints=0, segcls=0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.keypoints = keypoints
        self.segcls = segcls

    def __call__(self, image, targets, input_dim, seg_targets):
        boxes = targets[:, :4].copy()  # xyxy格式
        labels = targets[:, 4].copy()
        landmarks = targets[:, -3 * self.keypoints:].copy() if self.keypoints > 0 else np.array([])
        seg = seg_targets.copy() if self.segcls > 0 else np.array([])
        
        # 无标签的情况：直接预处理并返回空标签
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5 + 3 * self.keypoints), dtype=np.float32)
            if self.keypoints > 0:
                targets[..., -3 * self.keypoints:] = targets[..., -3 * self.keypoints:] * 0####- 1
            if self.segcls > 0:
                seg_targets = seg_targets * 0
            else:
                seg_targets = np.array([])
            image, seg_targets, r_o, dx_o, dy_o = preproc(image, input_dim, seg_targets)
            return image, targets, seg_targets

        # 保存原始图像和标签（用于回退）
        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]  # xyxy格式
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)  # 转为cxcywh格式
        landmarks_o = targets_o[:, -3 * self.keypoints:].copy() if self.keypoints > 0 else np.array([])
        segs_o = seg_targets.copy() if self.segcls > 0 else np.array([])

        # HSV色彩增强
        if random.random() < self.hsv_prob:
            augment_hsv(image)

        # 镜像翻转
        image_t, boxes, landmarks, seg_t = _mirror(image, boxes, landmarks, seg, self.flip_prob)
        height, width, _ = image_t.shape
        
        # Letterbox居中预处理（获取缩放比例r_和偏移dx/dy）
        image_t, seg_t, r_, dx, dy = preproc(image_t, input_dim, seg_t)
        
        # ===================== 核心修正1：处理翻转后标签的缩放+偏移 =====================
        # 1. 边界框：先转cxcywh → 缩放 → 加偏移 → 边界校验
        boxes = xyxy2cxcywh(boxes)  # xyxy → cxcywh
        boxes[:, 0] = boxes[:, 0] * r_ + dx  # cx = 原始cx×r + 水平偏移dx
        boxes[:, 1] = boxes[:, 1] * r_ + dy  # cy = 原始cy×r + 垂直偏移dy
        boxes[:, 2] = boxes[:, 2] * r_       # w = 原始w×r（宽高只缩放，无偏移）
        boxes[:, 3] = boxes[:, 3] * r_       # h = 原始h×r
        
        # 2. 关键点：缩放 → 加偏移（每个x加dx，每个y加dy）
        if self.keypoints > 0 and len(landmarks) > 0 and landmarks.size > 0:
            # landmarks.shape = (N, 12) → 结构：x/y/vis
            # 步骤1：缩放所有坐标
            landmarks[:, 0::3] *= r_  # 步长3，从0开始：所有x坐标列
            landmarks[:, 1::3] *= r_  
            # 步骤2：提取可见性mask（仅对vis>0的点偏移）
            # 2.1 提取vis列（2/5/8/11）→ shape: (N, 4)
            vis_cols = landmarks[:, 2::3] 
            # 2.2 生成可见性mask：vis>0 视为可见 
            vis_mask = (vis_cols > 0).astype(landmarks.dtype)  # 转为和landmarks同类型，避免类型错误
            # 步骤3：仅对可见的点执行x/y偏移
            # mask.shape=(N,4)，与x/y列（0::3/1::3）的shape完全匹配，可直接相乘过滤
            landmarks[:, 0::3] += dx * vis_mask  # x坐标：仅可见点加dx
            landmarks[:, 1::3] += dy * vis_mask  # y坐标：仅可见点加dy

        # 过滤无效框（宽/高>1）
        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        landmarks_t = landmarks[mask_b] if (self.keypoints > 0 and len(landmarks) > 0) else np.array([])

        # ===================== 核心修正2：回退逻辑中处理原始标签的缩放+偏移 =====================
        if len(boxes_t) == 0:
            # 用原始图像的标签回退，同样处理缩放+偏移
            image_, segs_ot, r_o, dx_o, dy_o = preproc(image_o, input_dim, segs_o)
            # 边界框：缩放 + 加偏移
            boxes_o[:, 0] = boxes_o[:, 0] * r_o + dx_o
            boxes_o[:, 1] = boxes_o[:, 1] * r_o + dy_o
            boxes_o[:, 2] = boxes_o[:, 2] * r_o
            boxes_o[:, 3] = boxes_o[:, 3] * r_o
            boxes_t = boxes_o
            labels_t = labels_o
            # 关键点：缩放 + 加偏移
            if self.keypoints > 0 and len(landmarks_o) > 0 and landmarks_o.size > 0:
                landmarks_o[:, 0::3] *= r_o    # x坐标加偏移
                landmarks_o[:, 1::3] *= r_o    # y坐标加偏移
                # 步骤2：提取可见性mask（仅对vis>0的点偏移）
                # 2.1 提取vis列（2/5/8/11）→ shape: (N, 4)
                vis_cols_o = landmarks_o[:, 2::3] 
                # 2.2 生成可见性mask：vis>0 视为可见 
                vis_mask_o = (vis_cols_o > 0).astype(landmarks_o.dtype)  # 转为和landmarks同类型，避免类型错误
                # 步骤3：仅对可见的点执行x/y偏移
                # mask.shape=(N,4)，与x/y列（0::3/1::3）的shape完全匹配，可直接相乘过滤
                landmarks_o[:, 0::3] += dx_o * vis_mask_o  # x坐标：仅可见点加dx
                landmarks_o[:, 1::3] += dy_o * vis_mask_o  # y坐标：仅可见点加dy
            landmarks_t = landmarks_o

        # 拼接标签并填充到max_labels长度
        labels_t = np.expand_dims(labels_t, 1)
        if self.keypoints > 0 and len(landmarks_t) > 0:
            targets_t = np.hstack((labels_t, boxes_t, landmarks_t))
            padded_labels = np.zeros((self.max_labels, 5 + self.keypoints * 3))
        else:
            targets_t = np.hstack((labels_t, boxes_t)) if len(boxes_t) > 0 else np.zeros((0,5))
            padded_labels = np.zeros((self.max_labels, 5))
        
        # 填充有效标签（避免越界）
        valid_len = min(len(targets_t), self.max_labels)
        padded_labels[:valid_len] = targets_t[:valid_len]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        
        return image_t, padded_labels, seg_t


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, seg, r, dx, dy = preproc(img, input_size, np.array([]), self.swap)
        # if self.legacy:  # F
        #     img = img[::-1, :, :].copy()
        #     img /= 255.0
        #     img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        #     img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5)), r, dx, dy
