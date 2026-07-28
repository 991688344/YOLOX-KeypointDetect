#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
QATEvalWrapper 单元测试：验证 QAT 模型能被 COCO 评估器接口正确消费。
不跑完整 COCO eval，只验证 wrapper 的 forward 输出格式与 decode 接口。

用法：
  CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/lyc_py38/bin/python tests/test_qat_eval_wrapper.py \
      -f exps/example/custom/yolox_kp.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from yolox.exp import get_exp
from yolox.utils.quant_utils import prepare_qat_model, QATEvalWrapper


def main():
    parser = argparse.ArgumentParser("QAT eval wrapper test")
    parser.add_argument("-f", "--exp_file", required=True)
    args = parser.parse_args()

    exp = get_exp(args.exp_file, None)
    model = exp.get_model()
    model.eval()

    print("[1/6] Preparing QAT model...")
    prepared = prepare_qat_model(model, exp, backend="qnnpack")
    print("    OK")

    # Keep an original head for the wrapper (mirrors trainer.qat_head)
    original_head = model.head
    wrapper = QATEvalWrapper(prepared, original_head)

    B = 2
    H, W = exp.test_size
    imgs = torch.randn(B, exp.img_channel, H, W)

    print("[2/6] Wrapper forward: expect [B, N, C] eval layout...")
    with torch.no_grad():
        outputs = wrapper(imgs)
    print(f"    output shape: {tuple(outputs.shape)}")

    # Expected dims for yolox_kp: C = 4 + 1 + num_classes + keypoints*3
    exp_C = 4 + 1 + exp.num_classes + exp.keypoints * 3
    exp_N = (H // 8) * (W // 8) + (H // 16) * (W // 16) + (H // 32) * (W // 32)
    assert outputs.shape == (B, exp_N, exp_C), \
        f"Expected ({B}, {exp_N}, {exp_C}), got {tuple(outputs.shape)}"
    print(f"    shape correct: N={exp_N}, C={exp_C}")

    print("[3/6] Verify obj/cls channels are in [0,1] (sigmoid'd)...")
    obj = outputs[..., 4]
    cls = outputs[..., 5:5 + exp.num_classes]
    assert (obj >= 0).all() and (obj <= 1).all(), "obj not sigmoid'd!"
    assert (cls >= 0).all() and (cls <= 1).all(), "cls not sigmoid'd!"
    print("    obj/cls sigmoid'd: OK")

    print("[4/6] Evaluator accesses model.head.decode_in_inference...")
    _ = wrapper.head.decode_in_inference
    print(f"    decode_in_inference = {wrapper.head.decode_in_inference}")

    print("[5/6] Evaluator calls model.head.decode_outputs (needs hw set)...")
    decoder = wrapper.head.decode_outputs
    decoded, _ = decoder(outputs.clone(), None, dtype=outputs.type())
    assert decoded.shape == (B, exp_N, exp_C), \
        f"decoded shape mismatch: {tuple(decoded.shape)}"
    # decode should scale cxcy/wh by stride -> values much larger than raw
    print(f"    decoded shape: {tuple(decoded.shape)}")
    print(f"    decoded bbox cxcy max: {decoded[..., :2].max().item():.1f} "
          f"(raw was {outputs[..., :2].max().item():.3f})")

    print("[6/6] adjust_status compatibility (eval mode toggle)...")
    from yolox.utils import adjust_status
    with adjust_status(wrapper, training=False):
        with torch.no_grad():
            out2 = wrapper(imgs)
    assert out2.shape == (B, exp_N, exp_C)
    print("    adjust_status works: OK")

    print("\n" + "=" * 60)
    print("EVAL WRAPPER TEST PASSED!")
    print(f"  - forward -> [B, N, C] = ({B}, {exp_N}, {exp_C}): OK")
    print(f"  - obj/cls sigmoid'd: OK")
    print(f"  - head.decode_in_inference accessible: OK")
    print(f"  - head.decode_outputs works (hw set): OK")
    print(f"  - adjust_status compatible: OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
