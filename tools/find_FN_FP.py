# 自动判断漏检误检的图像，并保存漏检误检的图像到指定文件夹中
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

def get_video_list(path):
    """返回文件夹下所有 .mp4 文件路径"""
    video_paths = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.mp4'):
                video_paths.append(os.path.join(root, f))
    return video_paths

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

        for i in range(len(bboxes)):
            cls_id = cls_ids[i]
            if draw_cls is not None and cls_id != draw_cls:
                continue

            x1, y1, x2, y2 = bboxes[i] + IMG_BOARDER
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = COLORS[(cls_ids[i]+10)%len(COLORS)].tolist()
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        return img


def analyze_detections(detected, total_frames, 
                    miss_max_gap_thresh=5,
                    miss_min_coverage_thresh = 0.2,
                    fp_max_coverage_thresh=0.15,
                    #    fp_max_len_ratio=0.1,
                    fp_max_len=20):
    """
    检测漏检和误检帧。
    
    参数:
        detected: bool list，长度 total_frames，表示每帧是否有该类别检测框
        total_frames: 视频总帧数
        miss_max_gap_thresh: 漏检的最大连续缺失帧数
        miss_min_coverage_thresh: 漏检覆盖率阈值（低于此值不予考虑漏检）
        fp_max_coverage_thresh: 误检覆盖率阈值（低于此值认为整体稀少）
        fp_max_len_ratio: 当覆盖率低时，单段检测区间超过总帧数的此比例则不视为误检
        fp_max_len: 当覆盖率低时，单段检测区间超过此固定帧数也不视为误检
    
    返回:
        false_negatives: list[int] 漏检帧号
        false_positives: list[int] 误检帧号
    """
    # 1. 提取检测区间（连续True的区间）
    intervals = []
    i = 0
    while i < total_frames:
        if detected[i]:
            start = i
            while i < total_frames and detected[i]:
                i += 1
            end = i - 1
            intervals.append([start, end])
        else:
            i += 1

    false_negatives = []
    false_positives = []
    # 2. 计算片段整体占比
    total_det_frames = sum(e - s + 1 for s, e in intervals)
    coverage = total_det_frames / total_frames

    # 3. 漏检： 覆盖率>20% && 相邻区间之间的短间隙 < miss_max_gap  （不包含首尾间隙）
    if coverage >= miss_min_coverage_thresh:
        for idx in range(len(intervals) - 1):
            gap_start = intervals[idx][1] + 1
            gap_end = intervals[idx+1][0] - 1
            if gap_start <= gap_end:
                gap_len = gap_end - gap_start + 1
                if gap_len <= miss_max_gap_thresh:
                    false_negatives.extend(range(gap_start, gap_end + 1))

    # 4. 误检：基于整体覆盖率判断
    # 动态阈值：优先使用比率阈值，但不超过固定最大长度
    # max_len_thresh = min(int(total_frames * fp_max_len_ratio), fp_max_len)
    max_len_thresh = fp_max_len
    if coverage < fp_max_coverage_thresh:
        # 整体稀疏，检查每个区间是否“短暂”
        for s, e in intervals:
            interval_len = e - s + 1
            if interval_len <= max_len_thresh:
                false_positives.extend(range(s, e + 1))
        # 如果有一个区间很长（超过阈值），则不标记该区间（可能为真实短暂出现）
    # else: 覆盖率足够高，不标记任何误检（认为出现的都是真实目标）
    
    return false_negatives, false_positives

def save_frames_from_video(video_path, frame_ids, output_dir, prefix=""):
    """
    从视频中提取指定帧号（0-indexed）的原始图像，保存到 output_dir 下。
    文件名: {prefix}_frame_{frame_id:05d}.png
    """
    if not frame_ids:
        return
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频 {video_path} 以保存帧")
        return
    # 按帧号从小到大排序并去重
    frame_ids = sorted(set(frame_ids))
    current_frame = 0
    for fid in frame_ids:
        # 跳到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            save_name = f"{prefix}_frame_{fid:05d}.png"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, frame)
        else:
            logger.warning(f"无法读取视频帧 {fid} 从 {video_path}")
    cap.release()


def process_one_video(video_path, predictor, args):
    """处理单个视频：可视化输出（若save_result）以及错误分析（若error_analysis）"""
    logger.info(f"处理视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_w = w + 2*IMG_BOARDER
    new_h = h + 2*IMG_BOARDER

    # 准备结果视频写入器
    vid_writer = None
    if args.save_result:
        os.makedirs(args.save_dir, exist_ok=True)
        video_name = os.path.basename(video_path)
        save_video_path = os.path.join(args.save_dir, video_name)
        vid_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_w, new_h))
        logger.info(f"结果视频将保存到: {save_video_path}")

    # 目标类别 ID
    target_cls = BBOX_CLASSES.index(args.draw_cls) if args.draw_cls is not None else None

    # 用于错误分析的检测记录
    if args.error_analysis:
        phone_detected = [False] * total_frames
        smoke_detected = [False] * total_frames

    pbar = tqdm(total=total_frames, desc="推理中")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outputs, _ = predictor.inference(frame)
        # 记录检测信息
        if args.error_analysis:
            detections = outputs[0]
            if detections is not None:
                detections = detections.cpu().numpy()
                cls_ids = detections[:, 6].astype(int)
                # 标记 Phone (4) 和 Smoking (3)
                if 4 in cls_ids:
                    phone_detected[frame_idx] = True
                if 3 in cls_ids:
                    smoke_detected[frame_idx] = True

        # 绘制可视化帧
        res_frame = predictor.visual(outputs[0], frame, draw_kp=args.keypoints, draw_cls=target_cls)
        if vid_writer is not None:
            vid_writer.write(res_frame)
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if vid_writer:
        vid_writer.release()

    # 错误分析：计算需保存的帧并提取
    if args.error_analysis:
        logger.info(f"正在进行错误分析: {video_path}")
        # Phone 漏检/误检
        phone_fn, phone_fp = analyze_detections(
            phone_detected, total_frames,
            miss_max_gap_thresh=args.miss_max_gap,
            miss_min_coverage_thresh=args.miss_min_coverage_thresh,
            fp_max_coverage_thresh=args.fp_max_coverage_thresh,
            # fp_max_len_ratio=args.fp_max_len_ratio,
            fp_max_len=args.fp_max_len
        )
        # Smoking 漏检/误检
        smoke_fn, smoke_fp = analyze_detections(
            smoke_detected, total_frames,
            miss_max_gap_thresh=args.miss_max_gap,
            miss_min_coverage_thresh=args.miss_min_coverage_thresh,
            fp_max_coverage_thresh=args.fp_max_coverage_thresh,
            # fp_max_len_ratio=args.fp_max_len_ratio,
            fp_max_len=args.fp_max_len
        )
        video_basename = os.path.splitext(os.path.basename(video_path))[0]

        # 保存漏检图像
        if phone_fn:
            loujian_phone_dir = os.path.join(args.loujian_root, "Phone")
            save_frames_from_video(video_path, phone_fn, loujian_phone_dir, prefix=f"{video_basename}_Phone_LouJian")
            logger.info(f"Phone 漏检 {len(phone_fn)} 帧 -> {loujian_phone_dir}")
        if smoke_fn:
            loujian_smoke_dir = os.path.join(args.loujian_root, "Smoke")
            save_frames_from_video(video_path, smoke_fn, loujian_smoke_dir, prefix=f"{video_basename}_Smoke_LouJian")
            logger.info(f"Smoke 漏检 {len(smoke_fn)} 帧 -> {loujian_smoke_dir}")

        # 保存误检图像
        if phone_fp:
            wujian_phone_dir = os.path.join(args.wujian_root, "Phone")
            save_frames_from_video(video_path, phone_fp, wujian_phone_dir, prefix=f"{video_basename}_Phone_WuJian")
            logger.info(f"Phone 误检 {len(phone_fp)} 帧 -> {wujian_phone_dir}")
        if smoke_fp:
            wujian_smoke_dir = os.path.join(args.wujian_root, "Smoke")
            save_frames_from_video(video_path, smoke_fp, wujian_smoke_dir, prefix=f"{video_basename}_Smoke_WuJian")
            logger.info(f"Smoke 误检 {len(smoke_fp)} 帧 -> {wujian_smoke_dir}")


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ONNX Infer with Error Analysis")
    parser.add_argument("--demo", default="image", help="image / video")
    parser.add_argument("--path", required=True, help="input path (image, video, or folder)")
    parser.add_argument("--onnx", required=True, help="onnx model path")
    parser.add_argument("--conf", default=0.3, type=float)
    parser.add_argument("--nms", default=0.3, type=float)
    parser.add_argument("--tsize", default=384, type=int)
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--save_dir", default="./output", help="output save directory for result videos")
    parser.add_argument("--save_result", action="store_true", help="save output video")
    parser.add_argument("--draw_cls", default=None, type=str, 
                        help="只绘制指定类别，可选：Person, Seatbelt, Face, Smoking, Phone")

    # 错误分析相关参数
    parser.add_argument("--error_analysis", action="store_true", help="perform false positive/negative analysis and save frames")
    parser.add_argument("--loujian_root", default="/home/liuyc/workspace/dataset/LouJian", help="root dir for false negatives (漏检)")
    parser.add_argument("--wujian_root", default="/home/liuyc/workspace/dataset/WuJian", help="root dir for false positives (误检)")
    parser.add_argument("--miss_max_gap", default=15, type=int, help="max gap (frames) for considering as miss (false negative)，超过15帧的漏检不予考虑")
    parser.add_argument("--miss_min_coverage_thresh", default=0.2, type=float, help="漏检覆盖率阈值，大于20%覆盖率的检测才进行漏检")
    parser.add_argument("--fp_max_coverage_thresh", default=0.2, type=float, help="误检覆盖率阈值，大于20%覆盖率的检测不视为误检")
    # parser.add_argument("--fp_max_len_ratio", default=0.1, type=float, help="误检单段最大长度占视频总帧数的比例")
    parser.add_argument("--fp_max_len", default=20, type=int, help="误检单段最大长度的固定帧数上限，大于20帧的检测区间不视为误检")
    return parser


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
        # 图像模式保持原有逻辑
        os.makedirs(args.save_dir, exist_ok=True)
        files = get_image_list(args.path) if os.path.isdir(args.path) else [args.path]
        target_cls = BBOX_CLASSES.index(args.draw_cls) if args.draw_cls is not None else None
        for img_path in files:
            outputs, img = predictor.inference(img_path)
            res_img = predictor.visual(outputs[0], img, draw_kp=args.keypoints, draw_cls=target_cls)
            if args.save_result:
                save_name = os.path.basename(img_path)
                save_path = os.path.join(args.save_dir, save_name)
                cv2.imwrite(save_path, res_img)
                logger.info(f"图片已保存：{save_path}")

    elif args.demo == "video":
        if os.path.isdir(args.path):
            video_list = get_video_list(args.path)
            total_videos = len(video_list)
            logger.info(f"找到 {total_videos} 个 mp4 视频文件")
            for idx, vpath in enumerate(video_list, start=1):
                percent = idx / total_videos * 100
                logger.info(f"正在处理视频 [{idx}/{total_videos}] ({percent:.1f}%): {vpath}")
                process_one_video(vpath, predictor, args)
        else:
            process_one_video(args.path, predictor, args)

if __name__ == "__main__":
    main()