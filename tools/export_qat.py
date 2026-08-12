#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Export QAT-trained model to ONNX with QDQ nodes for RKNN deployment.

Usage (from saved QAT checkpoint):
  conda activate lyc_py38
  python tools/export_qat.py \
      -f exps/example/custom/yolox_kp.py \
      -c YOLOX_outputs/yolox_kp_XXXXXX/best_ckpt.pth \
      --output-name weights/onnx/yoloxn_qat.onnx \
      --opset 19

Usage (from pretrained float ckpt, for QAT fine-tuning + export):
  python tools/export_qat.py \
      -f exps/example/custom/yolox_kp.py \
      -c weights/pt/yoloxn_DWFT_pre_784.pth \
      --output-name weights/onnx/yoloxn_qat.onnx \
      --opset 19 --finetune_epochs 10 --lr 0.0001

After export, convert to RKNN:
  conda activate rknn2.3.2
  python /home/liuyc/workspace/programmers/tools/onnx2rknn2_qat.py \
      --onnx_model weights/onnx/yoloxn_qat.onnx \
      --out_path weights/rknn/yoloxn_qat.rknn
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from loguru import logger
from yolox.exp import get_exp
from yolox.utils.quant_utils import (
    prepare_qat_model,
    compute_qat_loss,
    convert_and_export_qat,
    TraceableYOLOXForQAT,
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX QAT ONNX export")
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True)
    parser.add_argument("-c", "--ckpt", default=None, type=str, required=True,
                        help="Checkpoint path (QAT-trained or pretrained float)")
    parser.add_argument("--output-name", default="yolox_qat.onnx", type=str)
    parser.add_argument("--opset", default=19, type=int)
    parser.add_argument("--backend", default="qnnpack", type=str,
                        choices=["qnnpack", "fbgemm"])
    parser.add_argument("--qat-weight-quant", dest="qat_weight_quant",
                        default="per_channel", type=str,
                        choices=["per_channel", "per_tensor"],
                        help="Weight quantization granularity. Must MATCH the "
                             "setting used during training. 'per_tensor' for "
                             "opset-12 (RV1126) export.")
    parser.add_argument("--no-onnxsim", action="store_true", default=True)
    parser.add_argument("--onnxsim", action="store_true", default=False,
                        help="Enable onnxsim (may break QDQ nodes)")
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)

    logger.info(f"Building model (num_classes={exp.num_classes}, "
                f"keypoints={exp.keypoints}, act={exp.act})...")
    model = exp.get_model()

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        ckpt_state = ckpt["model"]
    else:
        ckpt_state = ckpt

    # Try loading into the base model (float weights)
    # The filter-based load handles key mismatches from GraphModule vs YOLOX
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in ckpt_state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(filtered, strict=False)
    n_loaded = len(filtered)
    n_skipped = len(ckpt_state) - n_loaded
    logger.info(f"Loaded {n_loaded} params, skipped {n_skipped} "
                f"(fake-quant buffers from QAT ckpt are expected to skip)")

    # Prepare QAT model (insert fake-quant)
    model.eval()
    prepared = prepare_qat_model(model, exp, backend=args.backend,
                                 weight_quant=args.qat_weight_quant)

    # If the ckpt is a QAT-trained checkpoint (has fake-quant keys),
    # load them into the prepared model
    prepared_state = prepared.state_dict()
    qat_filtered = {
        k: v for k, v in ckpt_state.items()
        if k in prepared_state and v.shape == prepared_state[k].shape
    }
    if len(qat_filtered) > n_loaded:
        prepared.load_state_dict(qat_filtered, strict=False)
        logger.info(f"Loaded {len(qat_filtered) - n_loaded} additional "
                    f"fake-quant params from QAT checkpoint")

    # Export ONNX with QDQ nodes
    convert_and_export_qat(
        prepared, exp,
        output_name=args.output_name,
        opset=args.opset,
        no_onnxsim=not args.onnxsim,
    )

    logger.info(f"Done! QAT ONNX saved to: {args.output_name}")
    logger.info(f"Next: convert to RKNN with onnx2rknn2_qat.py")


if __name__ == "__main__":
    main()
