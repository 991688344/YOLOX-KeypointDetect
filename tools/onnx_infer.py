#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from tqdm import tqdm
import torch
import torchvision

# 常量定义
BBOX_CLASSES = (
    "Person",
    "Seatbelt",
    "Face",
    "Smoking",
    "Phone"
)
IMG_BOARDER = 100
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4"]  # 新增：支持的视频格式
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

def decode_outputs(outputs, num_classes=5):
    predictions = []
    for i in range(0, len(outputs), 4):
        reg = outputs[i]
        obj = outputs[i+1]
        cls = outputs[i+2]
        kpt = outputs[i+3]

        reg = torch.from_numpy(reg)
        obj = torch.from_numpy(obj)
        cls = torch.from_numpy(cls)
        kpt = torch.from_numpy(kpt)

        B, C, H, W = reg.shape
        reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
        obj = obj.permute(0, 2, 3, 1).reshape(B, -1, 1)
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, num_classes)
        kpt = kpt.permute(0, 2, 3, 1).reshape(B, -1, 12)

        shift_x = torch.arange(0, W).repeat(H, 1)
        shift_y = torch.arange(0, H).repeat(W, 1).t()
        shift = torch.stack([shift_x, shift_y], dim=2).float().unsqueeze(0)
        shift = shift.reshape(1, -1, 2)

        stride = 8 * (2 ** (i//4))
        cxcy = (reg[..., :2] + shift) * stride
        wh = reg[..., 2:].exp() * stride
        bbox = torch.cat([cxcy, wh], dim=-1)

        pred = torch.cat([bbox, obj.sigmoid(), cls.sigmoid(), kpt], dim=-1)
        predictions.append(pred)

    return torch.cat(predictions, dim=1).numpy()

def postprocess(prediction, letterbox_info, num_classes, conf_thre=0.7, nms_thre=0.7,
                class_agnostic=False, keypoints=False, segs=False):
    box_corner = np.copy(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    img_h = letterbox_info["img_h"]
    img_w = letterbox_info["img_w"]
    r = letterbox_info["r"]
    dx = letterbox_info["dx"]
    dy = letterbox_info["dy"]

    prediction = torch.from_numpy(prediction)
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        
        if keypoints or segs:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5+num_classes:]), 1)
        else:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
        else:
            nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)
        detections = detections[nms_out_index]
        if not detections.size(0):
            continue

        detections[:, 0] = (detections[:, 0] - dx) / r
        detections[:, 1] = (detections[:, 1] - dy) / r
        detections[:, 2] = (detections[:, 2] - dx) / r
        detections[:, 3] = (detections[:, 3] - dy) / r

        if keypoints and detections.shape[1] >= 7 + 12:
            kpts = detections[:, 7:19]
            kpts[:, 0::3] = (kpts[:, 0::3] - dx) / r
            kpts[:, 1::3] = (kpts[:, 1::3] - dy) / r
            kpts[:, 0::3] = torch.clamp(kpts[:, 0::3], 0, img_w)
            kpts[:, 1::3] = torch.clamp(kpts[:, 1::3], 0, img_h)
            detections[:, 7:19] = kpts

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
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], c), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / h, input_size[1] / w)
    resized_h, resized_w = int(h * r), int(w * r)
    resized_img = cv2.resize(
        img,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    dy = (input_size[0] - resized_h) // 2
    dx = (input_size[1] - resized_w) // 2
    padded_img[dy:dy + resized_h, dx:dx + resized_w] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r, dx, dy

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

# 新增：获取目录下所有MP4视频文件
def get_video_list(path):
    video_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1].lower()  # 统一转小写，兼容.MP4/.mp4
            if ext in VIDEO_EXT:
                video_names.append(apath)
    return video_names

class ONNXPredictor:
    def __init__(self, onnx_path, input_size=(384, 640), num_classes=5, conf_thre=0.3, nms_thre=0.3, keypoints=False):
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

    def visual(self, output, img, draw_kp=False, draw_cls=None):
        h, w, _ = img.shape
        img = cv2.copyMakeBorder(img, IMG_BOARDER, IMG_BOARDER, IMG_BOARDER, IMG_BOARDER, cv2.BORDER_CONSTANT, value=[0,0,0])
        new_h, new_w = h + 2*IMG_BOARDER, w + 2*IMG_BOARDER

        if output is None:
            return img
        output = output.cpu().numpy()
        bboxes = output[:, 0:4]
        cls_ids = output[:, 6].astype(int)
        scores = output[:, 4] * output[:, 5]
        kps_list = output[:, 7:] if draw_kp else None

        valid_mask = scores >= self.conf_thre
        bboxes, cls_ids, scores = bboxes[valid_mask], cls_ids[valid_mask], scores[valid_mask]
        if draw_kp:
            kps_list = kps_list[valid_mask].reshape(-1, kps_list.shape[1]//3, 3)

        # ====================== 过滤：只绘制指定类别 ======================
        for i in range(len(bboxes)):
            cls_id = cls_ids[i]
            # 如果指定了 draw_cls，只画匹配的类别
            if draw_cls is not None and cls_id != draw_cls:
                continue

            x1, y1, x2, y2 = bboxes[i] + IMG_BOARDER
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = COLORS[(cls_ids[i]+10)%len(COLORS)].tolist()
            # 绘制检测框
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            
            # ====================== 新增：绘制检测类别名称 ======================
            cls_name = BBOX_CLASSES[cls_id]  # 获取类别名称
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6  # 文字大小
            thickness = 2     # 文字粗细
            # 文字位置：框的左上角上方
            text_pos = (x1, y1 - 10)
            # 绘制类别文字
            cv2.putText(img, cls_name, text_pos, font, font_scale, color, thickness)

        return img

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX Infer")
    parser.add_argument("--demo", default="image", help="image / video")
    parser.add_argument("--path", required=True, help="input path (file or dir)")
    parser.add_argument("--onnx", required=True, help="onnx model path")
    parser.add_argument("--conf", default=0.3, type=float)
    parser.add_argument("--nms", default=0.3, type=float)
    parser.add_argument("--tsize", default=384, type=int)
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--save_dir", default="./output", help="output save directory")
    parser.add_argument("--save_result", action="store_true", help="save output")
    
    # ✅ 新增：指定只绘制的类别
    parser.add_argument("--draw_cls", default=None, type=str, 
                        help="只绘制指定类别，可选：Person, Seatbelt, Face, Smoking, Phone")
    return parser

def image_demo(predictor, args):
    os.makedirs(args.save_dir, exist_ok=True)
    files = get_image_list(args.path) if os.path.isdir(args.path) else [args.path]
    
    # 解析要绘制的类别 ID
    target_cls = BBOX_CLASSES.index(args.draw_cls) if args.draw_cls is not None else None
    
    for img_path in files:
        outputs, img = predictor.inference(img_path)
        res_img = predictor.visual(outputs[0], img, draw_kp=args.keypoints, draw_cls=target_cls)
        if args.save_result:
            save_name = os.path.basename(img_path)
            save_path = os.path.join(args.save_dir, save_name)
            cv2.imwrite(save_path, res_img)
            logger.info(f"图片已保存：{save_path}")

# 重构：支持批量视频推理
def video_demo(predictor, args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 获取所有待处理的视频文件（目录/单个文件兼容）
    if os.path.isdir(args.path):
        video_files = get_video_list(args.path)
        logger.info(f"检测到目录，共找到 {len(video_files)} 个MP4视频")
    else:
        video_files = [args.path]
        logger.info(f"处理单个视频文件")

    if not video_files:
        logger.error("未找到任何MP4视频文件！")
        return

    # 解析要绘制的类别 ID
    target_cls = BBOX_CLASSES.index(args.draw_cls) if args.draw_cls is not None else None

    # 2. 遍历所有视频，依次推理
    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"[{idx}/{len(video_files)}] 开始处理视频：{video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"视频打开失败：{video_path}，跳过该视频")
            continue

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_w = w + 2*IMG_BOARDER
        new_h = h + 2*IMG_BOARDER
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 输出视频路径
        video_name = os.path.basename(video_path)
        save_video_path = os.path.join(args.save_dir, video_name)
        vid_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_w, new_h))

        # 逐帧处理
        pbar = tqdm(total=total_frames, desc=f"处理第{idx}个视频")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            outputs, _ = predictor.inference(frame)
            res_frame = predictor.visual(outputs[0], frame, draw_kp=args.keypoints, draw_cls=target_cls)
            vid_writer.write(res_frame)
            pbar.update(1)

        # 释放资源
        pbar.close()
        cap.release()
        vid_writer.release()
        logger.info(f"[{idx}/{len(video_files)}] 视频处理完成：{save_video_path}")

    logger.info("✅ 所有视频批量处理完成！")

def main():
    args = make_parser().parse_args()
    predictor = ONNXPredictor(
        onnx_path=args.onnx,
        input_size=(384, 640),
        conf_thre=args.conf,
        nms_thre=args.nms,
        keypoints=args.keypoints
    )

    if args.demo == "image":
        image_demo(predictor, args)
    elif args.demo == "video":
        video_demo(predictor, args)

if __name__ == "__main__":
    main()