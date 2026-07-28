#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
QAT 门槛测试：验证 FX graph-mode QAT 能否成功 trace YOLOX 推理路径。
这是整个 QAT 方案的生死关口。

用法：
  cd /home/liuyc/workspace/programmers/keypoint/yolox_ks
  conda activate lyc_py38
  python tests/test_qat_trace.py -f exps/example/custom/yolox_kp.py \
      [-c weights/pt/yoloxn_DWFT_pre_784.pth] [--opset 19]
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from yolox.exp import get_exp
from yolox.utils.quant_utils import (
    TraceableYOLOXForQAT,
    prepare_qat_model,
    compute_qat_loss,
    convert_and_export_qat,
    get_qat_qconfig,
    _disable_inplace_act,
)


def main():
    parser = argparse.ArgumentParser("QAT gate test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("--opset", default=19, type=int)
    parser.add_argument("--backend", default="qnnpack", type=str,
                        choices=["qnnpack", "fbgemm"])
    args = parser.parse_args()

    exp = get_exp(args.exp_file, None)
    print(f"[1/7] Exp loaded: num_classes={exp.num_classes}, keypoints={exp.keypoints}, "
          f"act={exp.act}, test_size={exp.test_size}, img_channel={exp.img_channel}")

    model = exp.get_model()
    print(f"[2/7] Model built: {type(model).__name__}")

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=True)
        print(f"    Loaded checkpoint: {args.ckpt}")

    model.eval()

    # ---- Step 3: FX trace test ----
    print("[3/7] Testing FX symbolic trace (before prepare)...")
    wrapper = TraceableYOLOXForQAT(model)
    _disable_inplace_act(model)
    wrapper.eval()

    example_input = torch.randn(1, exp.img_channel, exp.test_size[0], exp.test_size[1])

    # Test raw forward first
    with torch.no_grad():
        raw_out = wrapper(example_input)
    print(f"    Raw forward OK: {len(raw_out)} outputs, "
          f"shapes: {[o.shape for o in raw_out]}")

    # Test FX trace
    try:
        from torch.fx import symbolic_trace
        traced = symbolic_trace(wrapper)
        print(f"    FX symbolic_trace OK: {len(list(traced.graph.nodes))} graph nodes")
    except Exception as e:
        print(f"    FX symbolic_trace FAILED: {e}")
        print("    >>> This is the gate failure. Need custom Tracer or concrete_args.")
        # Try with concrete_args
        try:
            traced = symbolic_trace(wrapper, concrete_args={})
            print(f"    FX trace with concrete_args OK: {len(list(traced.graph.nodes))} nodes")
        except Exception as e2:
            print(f"    Also failed with concrete_args: {e2}")
            return

    # ---- Step 4: prepare_qat_fx test ----
    print("[4/7] Testing prepare_qat_fx...")
    prepared = prepare_qat_model(model, exp, backend=args.backend)
    print(f"    prepare_qat_fx OK, model type: {type(prepared).__name__}")

    # Verify forward through prepared model
    prepared.eval()
    with torch.no_grad():
        qat_out = prepared(example_input)
    print(f"    Prepared forward OK: {len(qat_out)} outputs")

    # Check output shapes match
    for i, (orig, qat) in enumerate(zip(raw_out, qat_out)):
        assert orig.shape == qat.shape, f"Output {i} shape mismatch: {orig.shape} vs {qat.shape}"
    print("    Output shapes match original: OK")

    # Check FakeQuantize modules
    from torch.ao.quantization import FakeQuantize
    n_fq = sum(1 for m in prepared.modules() if isinstance(m, FakeQuantize))
    print(f"    FakeQuantize modules: {n_fq}")
    assert n_fq > 0, "No FakeQuantize modules! QAT preparation failed."

    # ---- Step 5: Loss computation + backward test (BEFORE convert, matches real training) ----
    print("[5/7] Testing QAT loss computation + backward...")
    prepared.train()
    B = 2
    fake_imgs = torch.randn(B, exp.img_channel, exp.test_size[0], exp.test_size[1])
    # Create minimal fake targets: [B, max_labels, n_label_dim]
    # label layout: [class(1), bbox(4), keypoints(keypoints*3)] = 1+4+12=17 for 4 keypoints
    n_label_dim = 1 + 4 + exp.keypoints * 3
    max_labels = 10
    fake_targets = torch.zeros(B, max_labels, n_label_dim)
    # Add 2 fake objects per image
    for b in range(B):
        for obj_idx in range(2):
            fake_targets[b, obj_idx, 0] = 0  # class 0
            fake_targets[b, obj_idx, 1:5] = torch.tensor([100.0, 100.0, 50.0, 50.0])  # cx,cy,w,h
            # keypoints: x,y,vis for each keypoint
            kp_base = 5
            for kp in range(exp.keypoints):
                fake_targets[b, obj_idx, kp_base + kp*3] = 100.0 + kp*5  # x
                fake_targets[b, obj_idx, kp_base + kp*3 + 1] = 100.0 + kp*5  # y
                fake_targets[b, obj_idx, kp_base + kp*3 + 2] = 1.0  # vis

    raw_out_train = prepared(fake_imgs)
    loss_dict = compute_qat_loss(model.head, raw_out_train, fake_imgs, fake_targets)
    print(f"    Loss computed: total_loss={loss_dict['total_loss'].item():.4f}")

    # Test backward
    loss_dict['total_loss'].backward()
    print("    Backward pass OK")

    # Check that conv weights got gradients (prepared GraphModule shares weights with model)
    grad_count = 0
    for name, param in prepared.named_parameters():
        if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
            grad_count += 1
    print(f"    Prepared-model params with non-zero gradients: {grad_count}")
    assert grad_count > 0, "No gradients flowed! QAT training won't work."

    # ---- Step 6: convert_fx + export test (done AFTER training, once) ----
    print("[6/7] Testing convert_fx + export...")
    quantized = convert_and_export_qat(
        prepared, exp,
        output_name=os.path.join(tempfile.gettempdir(), "qat_test.onnx"),
        opset=args.opset,
        no_onnxsim=True,
    )
    print(f"    convert_fx + export OK: {quantized}")

    # ---- Step 7: Verify QDQ nodes in ONNX ----
    print("[7/7] Verifying QDQ nodes in exported ONNX...")
    import onnx
    onnx_model = onnx.load(os.path.join(tempfile.gettempdir(), "qat_test.onnx"))
    op_types = {}
    for node in onnx_model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    qdq_count = op_types.get('QuantizeLinear', 0) + op_types.get('DequantizeLinear', 0)
    n_outputs = len(onnx_model.graph.output)
    print(f"    ONNX ops: {dict(sorted(op_types.items(), key=lambda x: -x[1])[:10])}")
    print(f"    QDQ nodes: {qdq_count}")
    print(f"    Outputs: {n_outputs}")
    assert qdq_count > 0, "No QDQ nodes! Export won't work for RKNN."

    print("\n" + "=" * 60)
    print("ALL GATE TESTS PASSED! QAT is feasible for this model.")
    print(f"  - FX trace: OK ({n_fq} fake-quant modules)")
    print(f"  - Forward + loss + backward: OK ({grad_count} grads)")
    print(f"  - ONNX export: OK ({qdq_count} QDQ nodes, {n_outputs} outputs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
