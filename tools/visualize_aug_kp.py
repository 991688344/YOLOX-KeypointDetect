#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX 关键点检测训练数据增强可视化脚本。

从训练集随机挑选图片，走与训练完全一致的增强管线
(COCODataset.pull_item -> MosaicDetection/random_perspective -> TrainTransform(HSV/翻转/letterbox/噪声))，
并排展示"原图 | 增强图"，重点验证新加入的高斯/椒盐噪声增强（不含仿射）对真实训练图的影响。

输出目录：YOLOX_outputs/debug/
  overview.png         总览网格：每格 左=原图(letterbox+标签) | 右=完整 TrainTransform 增强图
                       （噪声按真实训练概率 0.5/0.3 随机触发，与训练完全一致）
  noise_ablation.png   单张逐项噪声对比（同一张 letterbox 底图，p=1.0 强制生效，隔离效果）：
                       Original | +Gaussian | +SaltPepper | +Gauss+Salt
  mosaic.png           Mosaic 完整管线样本（拼接+透视/仿射+HSV+letterbox+噪声），
                       验证 mosaic 路径下噪声同样生效（mosaic_prob 可视化时强制 1.0，真实训练 0.4）

用法：
  cd /home/liuyc/workspace/programmers/keypoint/yolox_ks
  /opt/miniconda3/envs/lyc_py38/bin/python tools/visualize_aug_kp.py
"""

import os
import sys
import random
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 项目根目录（脚本位于 tools/ 下），chdir 使相对路径 datasets/... 生效
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ_ROOT)
sys.path.insert(0, PROJ_ROOT)

from yolox.data import COCODataset, MosaicDetection, TrainTransform
from yolox.data.data_augment import preproc, add_gaussian_noise, add_salt_pepper
from yolox.exp import get_exp


def parse_opt():
    parser = argparse.ArgumentParser(description="Visualize YOLOX keypoint training data augmentation")
    parser.add_argument("--exp_file", type=str,
                        default=os.path.join(PROJ_ROOT, "exps", "example", "custom", "yolox_kp.py"),
                        help="exp 配置文件")
    parser.add_argument("--num_samples", type=int, default=8, help="总览样本数")
    parser.add_argument("--grid_cols", type=int, default=4, help="总览网格列数")
    parser.add_argument("--mosaic_samples", type=int, default=2, help="Mosaic 样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，固定便于复现")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJ_ROOT, "YOLOX_outputs", "debug"),
                        help="输出目录")
    return parser.parse_args()


# ---------- 工具函数 ----------

def chw_to_hwc_bgr(arr):
    """float32/uint8 CHW [0,255] -> HWC uint8 BGR（cv2 绘图用）。"""
    a = np.asarray(arr)
    if a.dtype == np.float32:
        a = np.clip(a, 0, 255).round().astype(np.uint8)
    return a.transpose(1, 2, 0).copy()


def draw_targets(img_bgr, labels, keypoints=0, is_cxcywh=False):
    """在 BGR 图（HWC uint8）上绘制检测框(绿)+关键点(红)。labels: (N, 5+3*kp)。"""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for row in labels:
        if is_cxcywh:
            cx, cy, bw, bh = row[1:5]
            if bw < 1 or bh < 1:
                continue
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            # 原始标签格式: [x1,y1,x2,y2, cls, kps...]，xyxy 在前 4 列
            x1, y1, x2, y2 = row[0:4]
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
        x1 = int(max(0, min(w - 1, x1))); y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2))); y2 = int(max(0, min(h - 1, y2)))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 1)
        if keypoints > 0:
            kps = row[5:].reshape(-1, 3)
            for x, y, v in kps:
                if v > 0:
                    px = int(max(0, min(w - 1, round(x))))
                    py = int(max(0, min(h - 1, round(y))))
                    cv2.circle(out, (px, py), 2, (0, 0, 255), -1)
    return out


def letterbox_targets(targets, r, dx, dy, keypoints=0):
    """把 pull_item 返回的标签（已按 input_size 等比缩放后的图空间）映射到 letterbox 后的图空间。

    与 TrainTransform 逻辑一致：bbox 全部缩放+偏移；关键点仅对可见(v>0)的点偏移，
    不可见点坐标保持 0（避免无意义的平移）。
    """
    t = targets.copy()
    t[:, :4] = t[:, :4] * r
    t[:, 0] += dx; t[:, 1] += dy; t[:, 2] += dx; t[:, 3] += dy
    if keypoints > 0:
        kps = t[:, 5:].reshape(-1, keypoints, 3)
        vis = (kps[:, :, 2] > 0).astype(t.dtype)
        kps[:, :, 0] = kps[:, :, 0] * r + dx * vis
        kps[:, :, 1] = kps[:, :, 1] * r + dy * vis
        t[:, 5:] = kps.reshape(-1, keypoints * 3)
    return t


def save_grid(cells, rows, cols, out_path, titles=None, suptitle=None, dpi=100):
    """cells: list of HWC uint8 RGB 等尺寸图像，排成 rows*cols 网格。"""
    h, w = cells[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * w / dpi, rows * h / dpi))
    axes = np.array(axes).reshape(rows, cols)
    for i, cell in enumerate(cells):
        ax = axes[i // cols, i % cols]
        ax.imshow(cell)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    for j in range(len(cells), rows * cols):
        axes[j // cols, j % cols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if suptitle else [0, 0, 1, 1])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------- 可视化 ----------

def make_overview(dataset, exp, train_tt, idxs, grid_cols, out_path):
    cells, titles = [], []
    for i, idx in enumerate(idxs):
        random.seed(0x1F + i * 31)
        np.random.seed(0x1F + i * 31)
        img, targets, _, _, _ = dataset.pull_item(idx)

        # 左：原图 letterbox + 原始标签
        base, _, r, dx, dy = preproc(img, exp.input_size, np.array([]))
        orig_bgr = draw_targets(
            chw_to_hwc_bgr(base),
            letterbox_targets(targets, r, dx, dy, exp.keypoints),
            exp.keypoints)
        # 右：完整 TrainTransform（含噪声，按真实概率触发，与训练一致）
        aug_img, aug_labels, _ = train_tt(img, targets, exp.input_size, np.array([]))
        aug_bgr = draw_targets(
            chw_to_hwc_bgr(aug_img), aug_labels, exp.keypoints, is_cxcywh=True)

        cells.append(np.concatenate([orig_bgr, aug_bgr], axis=1)[..., ::-1])
        fname = dataset.annotations[idx][3] if idx < len(dataset.annotations) else ""
        titles.append(f"idx={idx} {os.path.basename(fname)}")

    rows = (len(cells) + grid_cols - 1) // grid_cols
    save_grid(cells, rows, grid_cols, out_path, titles=titles,
              suptitle="Left: original(letterbox) | Right: full TrainTransform "
                       f"(HSV+flip+letterbox+noise p={exp.noise_gauss_prob}/{exp.noise_sp_prob})")


def make_noise_ablation(dataset, exp, idx, out_path):
    random.seed(0x2A)
    np.random.seed(0x2A)
    img, targets, _, _, _ = dataset.pull_item(idx)
    base, _, r, dx, dy = preproc(img, exp.input_size, np.array([]))
    labels = letterbox_targets(targets, r, dx, dy, exp.keypoints)

    sigma = exp.noise_gauss_sigma
    p = exp.noise_sp_p
    variants = [
        ("Original", base),
        ("+Gaussian", add_gaussian_noise(base, sigma)),
        ("+SaltPepper", add_salt_pepper(base, p)),
        ("+Gauss+Salt", add_salt_pepper(add_gaussian_noise(base, sigma), p)),
    ]
    cells, titles = [], []
    for name, arr in variants:
        bgr = draw_targets(chw_to_hwc_bgr(arr), labels, exp.keypoints)
        cells.append(bgr[..., ::-1])
        titles.append(name)
    fname = dataset.annotations[idx][3] if idx < len(dataset.annotations) else ""
    save_grid(cells, 1, len(variants), out_path, titles=titles,
              suptitle=f"Noise ablation on {os.path.basename(fname)} "
                       f"(forced p=1.0; sigma={sigma}, sp_p={p}, [0,255] domain)")


def make_mosaic(dataset, exp, train_tt, idxs, out_path):
    mds = MosaicDetection(
        dataset,
        img_size=exp.input_size,
        mosaic=True,
        preproc=train_tt,
        degrees=exp.degrees,
        translate=exp.translate,
        mosaic_scale=exp.mosaic_scale,
        shear=exp.shear,
        perspective=exp.perspective,
        keypoints=exp.keypoints,
        segcls=exp.segcls,
        enable_mixup=exp.enable_mixup,
        mosaic_prob=1.0,   # 可视化时强制 mosaic 路径，真实训练为 exp.mosaic_prob
        mixup_prob=exp.mixup_prob,
    )
    cells, titles = [], []
    cols = 2
    for i, idx in enumerate(idxs):
        random.seed(0x3B + i * 17)
        np.random.seed(0x3B + i * 17)
        m_img, m_labels, _, _, _ = mds[idx]
        bgr = draw_targets(chw_to_hwc_bgr(m_img), m_labels, exp.keypoints, is_cxcywh=True)
        cells.append(bgr[..., ::-1])
        titles.append(f"mosaic sample idx={idx}")
    rows = (len(cells) + cols - 1) // cols
    save_grid(cells, rows, cols, out_path, titles=titles,
              suptitle="MosaicDetection full pipeline (mosaic+perspective+HSV+letterbox+noise), "
                       f"mosaic_prob forced=1.0 (real={exp.mosaic_prob})")


def main(opt):
    os.makedirs(opt.output, exist_ok=True)

    exp = get_exp(opt.exp_file, None)
    print(f"[exp] {os.path.basename(opt.exp_file)} | input_size={exp.input_size} | "
          f"keypoints={exp.keypoints} | mosaic_prob={exp.mosaic_prob} | "
          f"noise gauss p={exp.noise_gauss_prob} sigma={exp.noise_gauss_sigma} | "
          f"saltpepper p={exp.noise_sp_prob} sp_p={exp.noise_sp_p}")

    dataset = COCODataset(
        data_dir=exp.data_dir,
        json_file=exp.train_ann,
        img_size=exp.input_size,
        keypoints=exp.keypoints,
        segcls=exp.segcls,
        preproc=None,   # 手动 pull_item + 变换，便于并排展示
        cache=False,
    )
    print(f"[dataset] {len(dataset)} train images")

    train_tt = exp._train_transform(120)   # 与训练一致（含噪声概率 0.5/0.3）

    rng = random.Random(opt.seed)
    idxs = rng.sample(range(len(dataset)), min(opt.num_samples, len(dataset)))
    print(f"[overview] {len(idxs)} samples from train set")

    make_overview(dataset, exp, train_tt, idxs, opt.grid_cols,
                  os.path.join(opt.output, "overview.png"))
    make_noise_ablation(dataset, exp, idxs[0],
                        os.path.join(opt.output, "noise_ablation.png"))

    midx = rng.sample(range(len(dataset)), min(opt.mosaic_samples, len(dataset)))
    print(f"[mosaic] {len(midx)} samples (full MosaicDetection pipeline)")
    make_mosaic(dataset, exp, train_tt, midx, os.path.join(opt.output, "mosaic.png"))

    print(f"\n完成！请查看: {os.path.abspath(opt.output)}")
    print(f"  overview.png        总览：每格 左=原图 右=完整增强")
    print(f"  noise_ablation.png  噪声逐项对比（隔离效果）")
    print(f"  mosaic.png          Mosaic 完整管线样本")


if __name__ == "__main__":
    main(parse_opt())
