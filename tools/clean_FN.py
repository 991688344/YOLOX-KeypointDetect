#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
功能：对指定目录（如 /home/liuyc/workspace/dataset/LouJian/Phone）中的图片重新运行 ONNX 检测，
如果模型检测到了与 class_name 对应的目标（Phone 或 Smoking），则删除该图片。
也可通过目录名自动推断。
"""

import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from tqdm import tqdm
import torch
import torchvision

# -------------------------- 推理相关代码（与原脚本保持一致） --------------------------
BBOX_CLASSES = ("Person","Seatbelt","Face","Smoking","Phone")
CLASS_NAME_TO_ID = {"Phone": 4, "Smoking": 3}
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def decode_outputs(outputs, num_classes=5):
    predictions = []
    for i in range(0, len(outputs), 4):
        reg = outputs[i]; obj = outputs[i+1]; cls = outputs[i+2]; kpt = outputs[i+3]
        reg = torch.from_numpy(reg); obj = torch.from_numpy(obj); cls = torch.from_numpy(cls); kpt = torch.from_numpy(kpt)
        B, C, H, W = reg.shape
        reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
        obj = obj.permute(0, 2, 3, 1).reshape(B, -1, 1)
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, num_classes)
        kpt = kpt.permute(0, 2, 3, 1).reshape(B, -1, 12)
        shift_x = torch.arange(0, W).repeat(H, 1)
        shift_y = torch.arange(0, H).repeat(W, 1).t()
        shift = torch.stack([shift_x, shift_y], dim=2).float().unsqueeze(0).reshape(1, -1, 2)
        stride = 8 * (2 ** (i//4))
        cxcy = (reg[..., :2] + shift) * stride
        wh = reg[..., 2:].exp() * stride
        bbox = torch.cat([cxcy, wh], dim=-1)
        pred = torch.cat([bbox, obj.sigmoid(), cls.sigmoid(), kpt], dim=-1)
        predictions.append(pred)
    return torch.cat(predictions, dim=1).numpy()

def postprocess(prediction, letterbox_info, num_classes, conf_thre=0.3, nms_thre=0.3,
                class_agnostic=False, keypoints=False, segs=False):
    box_corner = np.copy(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    img_h = letterbox_info["img_h"]; img_w = letterbox_info["img_w"]
    r = letterbox_info["r"]; dx = letterbox_info["dx"]; dy = letterbox_info["dy"]
    prediction = torch.from_numpy(prediction)
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5:5+num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:,4]*detections[:,5], nms_thre)
        else:
            nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:,4]*detections[:,5], detections[:,6], nms_thre)
        detections = detections[nms_out_index]
        if not detections.size(0):
            continue
        detections[:, 0] = (detections[:, 0] - dx) / r
        detections[:, 1] = (detections[:, 1] - dy) / r
        detections[:, 2] = (detections[:, 2] - dx) / r
        detections[:, 3] = (detections[:, 3] - dy) / r
        detections[:, 0] = torch.clamp(detections[:, 0], 0, img_w)
        detections[:, 1] = torch.clamp(detections[:, 1], 0, img_h)
        detections[:, 2] = torch.clamp(detections[:, 2], 0, img_w)
        detections[:, 3] = torch.clamp(detections[:, 3], 0, img_h)
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    return output

def preprocess(img, input_size, swap=(2, 0, 1)):
    h, w, c = img.shape
    padded_img = np.ones((input_size[0], input_size[1], c), dtype=np.uint8) * 114
    r = min(input_size[0]/h, input_size[1]/w)
    resized_h, resized_w = int(h*r), int(w*r)
    resized_img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    dy = (input_size[0] - resized_h)//2
    dx = (input_size[1] - resized_w)//2
    padded_img[dy:dy+resized_h, dx:dx+resized_w] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r, dx, dy

class ONNXPredictor:
    def __init__(self, onnx_path, input_size=(384,640), num_classes=5, conf_thre=0.3, nms_thre=0.3, keypoints=False):
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.keypoints = keypoints

    def inference(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        height, width = img.shape[:2]
        img_processed, ratio, dx, dy = preprocess(img, self.input_size)
        img_processed = np.expand_dims(img_processed, axis=0)
        outputs = self.session.run(None, {self.input_name: img_processed})
        prediction = decode_outputs(outputs, self.num_classes)
        letterbox_info = {
            "img_h": height, "img_w": width, "r": ratio, "dx": dx, "dy": dy,
            "input_h": self.input_size[0], "input_w": self.input_size[1]
        }
        outputs = postprocess(prediction, letterbox_info, self.num_classes, self.conf_thre, self.nms_thre, False, self.keypoints, False)
        return outputs, img

# -------------------------- 业务逻辑 --------------------------
def get_image_files(folder):
    files = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in IMAGE_EXT:
            files.append(os.path.join(folder, f))
    return files

def process_folder(folder_path, target_class_name, predictor, dry_run=False):
    img_paths = get_image_files(folder_path)
    if not img_paths:
        logger.info(f"文件夹 {folder_path} 中没有图片文件")
        return
    target_id = CLASS_NAME_TO_ID[target_class_name]
    deleted = 0
    kept = 0
    logger.info(f"开始处理 {len(img_paths)} 张图片，目标类别: {target_class_name} (ID={target_id})")
    for img_path in tqdm(img_paths, desc=f"清理 {os.path.basename(folder_path)}"):
        outputs, _ = predictor.inference(img_path)
        dets = outputs[0]
        has_target = False
        if dets is not None:
            cls_ids = dets.cpu().numpy()[:, 6].astype(int)
            if target_id in cls_ids:
                has_target = True
        if has_target:
            if not dry_run:
                os.remove(img_path)
                logger.debug(f"删除: {img_path}")
            deleted += 1
        else:
            kept += 1
    logger.info(f"完成: 删除 {deleted} 张, 保留 {kept} 张" + (" (试运行)" if dry_run else ""))

def main():
    parser = argparse.ArgumentParser(description="清理漏检目录中实际包含目标的图片")
    parser.add_argument("--onnx", required=True, help="ONNX模型路径")
    parser.add_argument("--input_dir", required=True, help="要清理的图片目录，如 /home/liuyc/workspace/dataset/LouJian/Phone")
    parser.add_argument("--class_name", choices=["Phone", "Smoking"], 
                        help="手动指定类别名称 (Phone 或 Smoking)，若不指定则从目录名自动推断")
    parser.add_argument("--conf", default=0.3, type=float)
    parser.add_argument("--nms", default=0.3, type=float)
    parser.add_argument("--dry_run", action="store_true", help="试运行，不实际删除")
    args = parser.parse_args()

    # 确定类别名称
    if args.class_name is not None:
        class_name = args.class_name
    else:
        dir_name = os.path.basename(args.input_dir).lower()
        if "phone" in dir_name:
            class_name = "Phone"
        elif "smoke" in dir_name:
            class_name = "Smoking"
        else:
            logger.error(f"无法从目录名 '{dir_name}' 推断类别，请使用 --class_name 指定 (Phone 或 Smoking)")
            return

    predictor = ONNXPredictor(onnx_path=args.onnx, conf_thre=args.conf, nms_thre=args.nms)
    process_folder(args.input_dir, class_name, predictor, args.dry_run)

if __name__ == "__main__":
    main()