#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import glob
import shutil  # 新增：用于文件复制
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
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import torch.nn.functional as F
IMG_BOARDER = 100
from typing import Tuple

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".flv", ".mkv"]

# ====================== 新增：分类保存路径配置 ======================
HAS_PHONE_DIR = "/home/liuyc/workspace/dataset/hasPhoneCall"
NO_PHONE_DIR = "/home/liuyc/workspace/dataset/noPhoneCall"
HAS_SMOKE_DIR = "/home/liuyc/workspace/dataset/hasSmoke"
NO_SMOKE_DIR = "/home/liuyc/workspace/dataset/noSmoke"
# ====================================================================

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

def get_video_list(path):
    video_names = []
    if os.path.isdir(path):
        for ext in VIDEO_EXT:
            video_names.extend(glob.glob(os.path.join(path, f"*{ext}")))
            video_names.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXT:
            video_names.append(path)
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
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            if self.decoder is not None:
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
        return outputs, seg_output, img_info

    def visual(self, output, seg_output, img_info, cls_conf=0.35, draw_kp=False, draw_seg=False) -> Tuple[np.ndarray, np.ndarray]:
        ratio = img_info["ratio"]
        img = img_info["raw_img"].copy()
        h, w, _ = img.shape
        
        img = cv2.copyMakeBorder(
            img,
            top=IMG_BOARDER,
            bottom=IMG_BOARDER,
            left=IMG_BOARDER,
            right=IMG_BOARDER,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        new_h = h + 2 * IMG_BOARDER
        new_w = w + 2 * IMG_BOARDER
        seg_mask = np.zeros_like(img)

        if output is None:
            return img, seg_mask

        output = output.cpu()
        output_np = output.numpy()

        bboxes = output_np[:, 0:4] / ratio
        cls_ids = output_np[:, 6].astype(int)
        scores = output_np[:, 4] * output_np[:, 5]
        kps_list = output_np[:, 7:] / ratio if draw_kp else None

        valid_mask = scores >= cls_conf
        bboxes = bboxes[valid_mask]
        cls_ids = cls_ids[valid_mask]
        scores = scores[valid_mask]
        
        if draw_kp:
            kps_list = kps_list[valid_mask]
            kp_num = kps_list.shape[1] // 3
            kps_list = kps_list.reshape(-1, kp_num, 3)

        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            x1 += IMG_BOARDER
            y1 += IMG_BOARDER
            x2 += IMG_BOARDER
            y2 += IMG_BOARDER
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls_id = cls_ids[i]
            score = scores[i]
            cls_name = self.cls_names[cls_id]

            color = COLORS[(cls_id+10) % len(COLORS)].tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            
            text = f"{cls_name}: {score:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 0 else y1 + 20
            text_y = max(text_y, text_size[1] + 2)
            text_y = min(text_y, new_h - 2)
            cv2.rectangle(img, (text_x, text_y - text_size[1] - 2), 
                        (text_x + text_size[0], text_y + 2), color, -1)
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 255), thickness=1)

            if draw_kp and kps_list is not None:
                kps = kps_list[i]
                if cls_name in ['Face', 'Smoking', 'Phone']:
                    continue
                if cls_name == "Seatbelt":
                    kps = kps[:2]
                
                for idx, (kx, ky, visable) in enumerate(kps):
                    kx += IMG_BOARDER
                    ky += IMG_BOARDER
                    kx = int(kx)
                    ky = int(ky)
                    kp_radius = 6
                    
                    if 0 <= kx < new_w and 0 <= ky < new_h:
                        cv2.circle(img, (kx, ky), radius=kp_radius, color=color, thickness=-1)
                        
                        num_offset = 12
                        text_x = kx
                        text_y = ky - num_offset
                        if text_y < 0:
                            text_y = ky + num_offset
                        if text_x < 0:
                            text_x = 0
                        if text_x > new_w:
                            text_x = new_w
                        text_y = max(text_y, text_size[1] + 2)
                        text_y = min(text_y, new_h - 2)
                        
                        kp_num = str(idx)
                        text_size, _ = cv2.getTextSize(kp_num, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(img, (text_x - text_size[0]//2, text_y - text_size[1] - 2),
                                    (text_x + text_size[0]//2, text_y + 2), color, -1)
                        cv2.putText(img, kp_num, (text_x - text_size[0]//2, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1)

                kp_connections = {
                    "Person": [(0, 1), (1, 2), (1, 3)],
                    "Seatbelt": [(0, 1)]
                }
                if cls_name in kp_connections:
                    connections = kp_connections[cls_name]
                    current_kp_num = len(kps)
                    line_thickness = 2
                    for (p1_idx, p2_idx) in connections:
                        if p1_idx >= current_kp_num or p2_idx >= current_kp_num:
                            continue
                        kx1, ky1, vis1 = kps[p1_idx]
                        kx2, ky2, vis2 = kps[p2_idx]
                        kx1 += IMG_BOARDER
                        ky1 += IMG_BOARDER
                        kx2 += IMG_BOARDER
                        ky2 += IMG_BOARDER
                        kx1, ky1 = int(kx1), int(ky1)
                        kx2, ky2 = int(kx2), int(ky2)
                        if (0 <= kx1 < new_w and 0 <= ky1 < new_h) and (0 <= kx2 < new_w and 0 <= ky2 < new_h):
                            cv2.line(img, (kx1, ky1), (kx2, ky2), color, thickness=line_thickness)

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
            
            seg = cv2.copyMakeBorder(
                seg,
                top=IMG_BOARDER,
                bottom=IMG_BOARDER,
                left=IMG_BOARDER,
                right=IMG_BOARDER,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
            seg_mask = seg.astype(np.uint8)

        return img, seg_mask

# ====================== 核心修改：重写图片推理+分类函数 ======================
def image_demo(predictor, vis_folder, path, current_time, save_result, draw_kp, draw_seg):
    # 自动创建所有分类文件夹
    for dir_path in [HAS_PHONE_DIR, NO_PHONE_DIR, HAS_SMOKE_DIR, NO_SMOKE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图片
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()

    for image_name in files:
        # 1. 模型推理
        outputs, seg_outputs, img_info = predictor.inference(image_name)
        # 2. 初始化检测标志
        has_phone = False
        has_smoke = False

        # 3. 解析推理结果，判断是否包含Phone/Smoking
        if outputs[0] is not None:
            output = outputs[0].cpu()
            cls_ids = output[:, 6].cpu().numpy().astype(int)

            # 遍历所有检测目标
            for cls_id in cls_ids:
                cls_name = predictor.cls_names[cls_id]
                if cls_name == "Phone":
                    has_phone = True
                if cls_name == "Smoking":
                    has_smoke = True

        # 4. 按规则复制图片到对应文件夹
        img_filename = os.path.basename(image_name)
        # 复制Phone分类
        if has_phone:
            shutil.copy2(image_name, os.path.join(HAS_PHONE_DIR, img_filename))
            logger.info(f"✅ 检测到Phone: {img_filename} → 已复制到hasPhoneCall")
        else:
            shutil.copy2(image_name, os.path.join(NO_PHONE_DIR, img_filename))
            logger.info(f"❌ 未检测到Phone: {img_filename} → 已复制到noPhoneCall")
        # 复制Smoke分类
        if has_smoke:
            shutil.copy2(image_name, os.path.join(HAS_SMOKE_DIR, img_filename))
            logger.info(f"✅ 检测到Smoking: {img_filename} → 已复制到hasSmoke")
        else:
            shutil.copy2(image_name, os.path.join(NO_SMOKE_DIR, img_filename))
            logger.info(f"❌ 未检测到Smoking: {img_filename} → 已复制到noSmoke")

        # 保留原有的可视化保存功能（可选）
        if seg_outputs is None:
            seg_outputs = [None for _ in range(len(outputs))]
        result_image, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info, predictor.confthre, draw_kp, draw_seg)
        if save_result:
            save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)
# ============================================================================

def process_single_video(predictor, vis_folder, current_time, video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    new_width = orig_width + 2 * IMG_BOARDER
    new_height = orig_height + 2 * IMG_BOARDER
    
    vid_writer = None
    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, os.basename(video_path))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_width, new_height))
    
    pbar = tqdm(total=total_frames, desc=f"处理视频: {os.path.basename(video_path)}", unit="frame")
    
    try:
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                outputs, seg_outputs, img_info = predictor.inference(frame)
                if seg_outputs is None:
                    seg_outputs = [None for _ in range(len(outputs))]
                result_frame, seg_mask = predictor.visual(outputs[0], seg_outputs[0], img_info, predictor.confthre, draw_kp=True, draw_seg=False)
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
        logger.info(f"视频处理完成: {os.path.basename(video_path)}")

def imageflow_demo(predictor, vis_folder, current_time, args):
    video_list = get_video_list(args.path)
    if not video_list:
        logger.error(f"未找到视频文件！")
        return
    for video_path in video_list:
        process_single_video(predictor, vis_folder, current_time, video_path, args)

def crop(masks, boxes, padding=1, mask_ratio=4):
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

    exp.decode_in_inference = True
    model = exp.get_model()

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    if exp.model_name == 'yolov7_tiny':
        model.fuse()
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt if args.ckpt else os.path.join(file_name, "best_ckpt.pth")
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT不支持模型融合！"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "请先运行trt.py！"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
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