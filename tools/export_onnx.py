#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger
import sys
# activate rknn hack
if '--rknpu' in sys.argv:
    os.environ['RKNN_model_hack'] = '1'

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
print("current_dir:", current_dir)
# 获取祖父目录
grandparent_dir = os.path.dirname(current_dir)
print("grandparent_dir:", grandparent_dir)
sys.path.append(grandparent_dir)
import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument("--output-name", type=str, default="yolox.onnx", help="output name of models")
    parser.add_argument("--input", default="images", type=str, help="input node name of onnx model")
    parser.add_argument("--output", default="output", type=str, help="output node name of onnx model")
    parser.add_argument("-o", "--opset", default=12, type=int, help="onnx opset version")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size (used as fixed batch if dynamic=False)")
    # 👇 修复：默认关闭动态维度（RKNN 友好），加 --dynamic 才开启
    parser.add_argument("--dynamic", default=False, action="store_true", help="whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--decode_in_inference", action="store_true", default=True, help="decode in inference or not")
    parser.add_argument('--rknpu', action="store_true", help='RKNN npu platform')
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference
    model.head.model_export = True
    if exp.model_name == 'yolov7_tiny':
        model.fuse()

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, exp.img_channel, exp.test_size[0], exp.test_size[1])

    # 👇 保留动态/静态切换逻辑
    dynamic_axes = {args.input: {0: 'batch'}, args.output: {0: 'batch'}} if args.dynamic else None

    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # 👇 修复核心：静态模式强制固定输入形状，动态模式保持不变
        if args.dynamic:
            input_shapes = None
        else:
            # 静态模式：RKNN 必须固定输入形状，彻底消除动态维度
            input_shapes = {args.input: [args.batch_size, exp.img_channel, exp.test_size[0], exp.test_size[1]]}

        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model, overwrite_input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()