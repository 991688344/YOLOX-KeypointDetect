#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import glob  # 新增：用于遍历视频文件
from loguru import logger
from tqdm import tqdm  

import cv2
import sys
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print("current_dir:", current_dir)
grandparent_dir = os.path.dirname(current_dir)
print("grandparent_dir:", grandparent_dir)
sys.path.append(grandparent_dir)
import torch
import numpy as np
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import BBOX_CLASSES, PLATE_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, load_ckpt, postprocess, vis
import torch.nn.functional as F
IMG_BOARDER = 100
from typing import Tuple

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".flv", ".mkv"]  # 新增：支持的视频格式

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-kp", "--keypoints", default=False, help="draw keypoints",
                        action="store_true")
    parser.add_argument("-segs", default=False, help="draw segmentation",
                        action="store_true")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    # ===================== 新增：指定GPU ID参数 =====================
    parser.add_argument("--gpuid", type=int, default=0, 
                        help="specify gpu id to use, only valid when device is gpu")
    # ==============================================================
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1].lower()
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

# 新增：获取视频文件列表（支持目录/单个文件）
def get_video_list(path):
    video_names = []
    if os.path.isdir(path):
        # 遍历目录下所有视频文件
        for ext in VIDEO_EXT:
            video_names.extend(glob.glob(os.path.join(path, f"*{ext}")))
            video_names.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))  # 兼容大写后缀
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXT:
            video_names.append(path)
    # 去重+排序
    video_names = sorted(list(set(video_names)))
    return video_names

class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=BBOX_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
            keypoints=False,
            segs=False
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.img_channel = exp.img_channel
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.keypoints = keypoints
        self.segs = segs
        self.mask_ratio = 4 if len(exp.in_channels) == 3 else 2
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            if self.img_channel == 4:
                img = cv2.imread(img, -1)
            else:
                img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _, r, dx, dy = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            
            if self.decoder is not None:  # None
                outputs, seg_output = self.model(img)
                outputs, seg_output = self.decoder(outputs, seg_output, dtype=outputs.type())
            else:
                outputs = self.model(img)
                seg_output = None

            letterbox_info = {
                "img_h": height,
                "img_w": width,
                "r": r,
                "dx": dx,
                "dy": dy,
                "input_h": self.test_size[0],
                "input_w": self.test_size[1],
            }
            outputs = postprocess(
                outputs[0], letterbox_info, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=False, keypoints=self.keypoints, segs=self.segs
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, seg_output, img_info

    def visual(self, output, seg_output, img_info, cls_conf=0.35, draw_kp=False, draw_seg=False) -> Tuple[np.ndarray, np.ndarray]:
        ratio = img_info["ratio"]
        img = img_info["raw_img"].copy()  # 复制原始图像，720, 1280
        h, w, _ = img.shape
        
        # ===================== 核心修改1：填充图像边界 =====================
        # 对原始图像进行四周填充，填充颜色为黑色(0)，上下左右各填充IMG_BOARDER像素
        img = cv2.copyMakeBorder(
            img,
            top=IMG_BOARDER,
            bottom=IMG_BOARDER,
            left=IMG_BOARDER,
            right=IMG_BOARDER,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # 填充黑色背景
        )
        # 计算填充后的新图像尺寸
        new_h = h + 2 * IMG_BOARDER
        new_w = w + 2 * IMG_BOARDER
        # =================================================================
        
        # 初始化分割掩码（尺寸与填充后的图像一致）
        seg_mask = np.zeros_like(img)

        if output is None:
            return img, seg_mask

        output = output.cpu()
        output_np = output.numpy()  # 张量 → NumPy数组

        # 还原坐标到原始图像尺度（未填充前）
        bboxes = output_np[:, 0:4] / ratio          # [360, 720] --> [720, 1280]
        cls_ids = output_np[:, 6].astype(int)       
        scores = output_np[:, 4] * output_np[:, 5]  
        kps_list = output_np[:, 7:] / ratio if draw_kp else None  # 还原到原始图像尺度

        # 步骤1：过滤低置信度的检测框
        valid_mask = scores >= cls_conf
        bboxes = bboxes[valid_mask]
        cls_ids = cls_ids[valid_mask]
        scores = scores[valid_mask]
        
        if draw_kp:
            kps_list = kps_list[valid_mask]
            kp_num = kps_list.shape[1] // 3  # 计算每个目标的关键点数量
            kps_list = kps_list.reshape(-1, kp_num, 3)  # 重塑为[N, k, 3]

        # 后续逐框绘制逻辑
        for i in range(len(bboxes)):
            # ===================== 核心修改2：偏移检测框坐标 =====================
            # 原始坐标 + 填充偏移量（让检测框对应到填充后的图像位置）
            x1, y1, x2, y2 = bboxes[i]
            x1 += IMG_BOARDER
            y1 += IMG_BOARDER
            x2 += IMG_BOARDER
            y2 += IMG_BOARDER
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # =================================================================
            
            cls_id = cls_ids[i]
            score = scores[i]
            cls_name = self.cls_names[cls_id]

            # 获取该检测框的专属颜色
            color = COLORS[(cls_id+10) % len(COLORS)].tolist()

            # 绘制检测框（基于填充后的坐标）
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            
            # 绘制类别名称和置信度（适配填充后的图像边界）
            text = f"{cls_name}: {score:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 0 else y1 + 20
            # 确保文本不超出填充后图像的边界
            text_y = max(text_y, text_size[1] + 2)
            text_y = min(text_y, new_h - 2)
            cv2.rectangle(img, (text_x, text_y - text_size[1] - 2), 
                        (text_x + text_size[0], text_y + 2), color, -1)
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), thickness=1)

            # 绘制关键点
            if draw_kp and kps_list is not None:
                kps = kps_list[i]
                if cls_name in ['Face', 'Smoking', 'Phone']:  # 这三类不绘制关键点
                    continue
                if cls_name == "Seatbelt":  # 安全带只取前两点
                    kps = kps[:2]
                
                for idx, (kx, ky, visable) in enumerate(kps):
                    # ===================== 核心修改3：偏移关键点坐标 =====================
                    # 原始关键点坐标 + 填充偏移量
                    kx += IMG_BOARDER
                    ky += IMG_BOARDER
                    # =================================================================
                    
                    kx = int(kx)
                    ky = int(ky)
                    kp_radius = 6  # 关键点半径
                    
                    # ===================== 核心修改4：更新关键点边界判断 =====================
                    # 边界判断改为填充后的图像尺寸，不再限制原图像范围
                    if 0 <= kx < new_w and 0 <= ky < new_h:
                    # =================================================================
                        # 绘制放大后的关键点（实心圆）
                        cv2.circle(img, (kx, ky), radius=kp_radius, color=color, thickness=-1)
                        
                        # 绘制关键点编号（适配填充后的图像边界）
                        num_offset = 12  # 编号偏移量
                        text_x = kx
                        text_y = ky - num_offset
                        # 确保编号文本不超出填充后的图像边界
                        if text_y < 0:
                            text_y = ky + num_offset
                        if text_x < 0:
                            text_x = 0
                        if text_x > new_w:
                            text_x = new_w
                        text_y = max(text_y, text_size[1] + 2)
                        text_y = min(text_y, new_h - 2)
                        
                        # 关键点编号（从0开始，可改为idx+1从1开始）
                        kp_num = str(idx)
                        
                        # 绘制编号文本背景
                        text_size, _ = cv2.getTextSize(kp_num, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(img, (text_x - text_size[0]//2, text_y - text_size[1] - 2),
                                    (text_x + text_size[0]//2, text_y + 2), color, -1)
                        # 绘制编号文本（白色字体）
                        cv2.putText(img, kp_num, (text_x - text_size[0]//2, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)

                # 绘制关键点线段（适配填充后的坐标）
                kp_connections = {
                    "Person": [(0, 1), (1, 2), (1, 3)],  # Person的连接对
                    "Seatbelt": [(0, 1)]                  # Seatbelt的连接对
                }
                if cls_name in kp_connections:
                    connections = kp_connections[cls_name]
                    current_kp_num = len(kps)
                    line_thickness = 2
                    for (p1_idx, p2_idx) in connections:
                        if p1_idx >= current_kp_num or p2_idx >= current_kp_num:
                            continue
                        # 获取原始关键点坐标并偏移
                        kx1, ky1, vis1 = kps[p1_idx]
                        kx2, ky2, vis2 = kps[p2_idx]
                        kx1 += IMG_BOARDER
                        ky1 += IMG_BOARDER
                        kx2 += IMG_BOARDER
                        ky2 += IMG_BOARDER
                        # 转为整数
                        kx1, ky1 = int(kx1), int(ky1)
                        kx2, ky2 = int(kx2), int(ky2)
                        # 边界判断（填充后的尺寸）
                        if (0 <= kx1 < new_w and 0 <= ky1 < new_h) and (0 <= kx2 < new_w and 0 <= ky2 < new_h):
                            cv2.line(img, (kx1, ky1), (kx2, ky2), color, thickness=line_thickness)

        # 分割绘制逻辑（适配填充后的图像）
        if draw_seg and seg_output is not None:
            sh, sw = seg_output.shape[:2]
            masks = torch.sigmoid(torch.matmul(seg_output, output[:, 7:].t()))
            masks = crop(masks, torch.from_numpy(bboxes).clone() * ratio, mask_ratio=self.mask_ratio)
            masks = masks.permute(2, 0, 1).contiguous()
            seg = F.interpolate(masks.unsqueeze(0),
                                (int(sh * self.mask_ratio / ratio),
                                int(sw * self.mask_ratio / ratio)),
                                mode='bilinear',
                                align_crops=False).squeeze(0).gt_(0.5).cpu().numpy()
            seg = seg * (cls_ids + 1)[:, None, None]
            seg = seg.astype('int').sum(axis=0)
            
            # ===================== 核心修改5：填充分割掩码 =====================
            # 原始分割掩码尺寸是(h, w)，需要填充到(new_h, new_w)
            seg = cv2.copyMakeBorder(
                seg,
                top=IMG_BOARDER,
                bottom=IMG_BOARDER,
                left=IMG_BOARDER,
                right=IMG_BOARDER,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
            # 确保分割掩码尺寸与填充后的图像一致
            seg_mask = seg.astype(np.uint8)
            # =================================================================

        return img, seg_mask


def image_demo(predictor, vis_folder, path, current_time, save_result, draw_kp, draw_seg):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, seg_outputs, img_info = predictor.inference(image_name)
        if seg_outputs is None:
            seg_outputs = [None for _ in range(len(outputs))]
        result_image, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info, predictor.confthre, draw_kp, draw_seg)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
            if draw_seg:
                if '.jpg' in save_file_name:
                    cv2.imwrite(save_file_name.replace('.jpg', '_seg.jpg'), seg_mask)
                else:
                    cv2.imwrite(save_file_name.replace('.png', '_seg.png'), seg_mask)

# 新增：封装 单个视频处理 函数
def process_single_video(predictor, vis_folder, current_time, video_path, args):
    """处理单个视频文件"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频基础信息
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算填充后的新尺寸
    new_width = orig_width + 2 * IMG_BOARDER
    new_height = orig_height + 2 * IMG_BOARDER
    
    # 初始化视频保存
    vid_writer = None
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, os.path.basename(video_path))
        logger.info(f"视频保存路径: {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (new_width, new_height)
        )
    
    # 进度条
    pbar = tqdm(total=total_frames, desc=f"处理视频: {os.path.basename(video_path)}", unit="frame")
    
    try:
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                outputs, seg_outputs, img_info = predictor.inference(frame)
                if seg_outputs is None:
                    seg_outputs = [None for _ in range(len(outputs))]
                result_frame, seg_mask = predictor.visual(
                    outputs[0], seg_outputs[0], img_info, 
                    predictor.confthre, draw_kp=True, draw_seg=False
                )
                if vid_writer is not None:
                    vid_writer.write(result_frame)
                pbar.update(1)
            else:
                break
    
    except Exception as e:
        logger.error(f"视频帧处理出错: {e}")
        raise
    finally:
        pbar.close()
        cap.release()
        if vid_writer is not None:
            vid_writer.release()
        logger.info(f"视频处理完成: {os.path.basename(video_path)}，共处理 {pbar.n} 帧")

# 修改：视频主处理函数（支持目录+单个视频）
def imageflow_demo(predictor, vis_folder, current_time, args):
    # 获取所有视频文件列表
    video_list = get_video_list(args.path)
    
    if not video_list:
        logger.error(f"未在路径 {args.path} 中找到任何视频文件！")
        return
    
    logger.info(f"共找到 {len(video_list)} 个视频文件，开始依次推理...")
    
    # 遍历所有视频，逐个处理
    for idx, video_path in enumerate(video_list, 1):
        logger.info(f"\n===== 正在处理第 {idx}/{len(video_list)} 个视频: {video_path} =====")
        process_single_video(predictor, vis_folder, current_time, video_path, args)
    
    logger.info("===== 所有视频推理完成！ =====")


def crop(masks, boxes, padding=1, mask_ratio=4):
    # h, w = shape
    h, w, n = masks.size()
    box_corner = boxes.clone()
    box_corner[..., 0] /= w * mask_ratio
    box_corner[..., 2] /= w * mask_ratio
    box_corner[..., 1] /= h * mask_ratio
    box_corner[..., 3] /= h * mask_ratio

    x1, x2 = sanitize_coordinates(box_corner[:, 0], box_corner[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(box_corner[:, 1], box_corner[:, 3], h, padding)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()

def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    exp.decode_in_inference = True
    model = exp.get_model()

    # ===================== 新增：绑定指定GPU =====================
    if args.device == "gpu":
        torch.cuda.set_device(args.gpuid)  # 锁定使用的GPU
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    # ============================================================
    if exp.model_name == 'yolov7_tiny':
        model.fuse()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load_ckpt 按 name+shape 过滤后 strict=False 加载，兼容 QAT ckpt：
        # QAT ckpt 含额外的 *_fake_quant.* 参数，会被自动跳过，只加载浮点权重。
        model = load_ckpt(model, ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, BBOX_CLASSES, trt_file, decoder, args.device,
                          args.fp16, args.legacy, args.keypoints, args.segs)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result,
                   args.keypoints, args.segs)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)