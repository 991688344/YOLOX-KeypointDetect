#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
QAT 冒烟测试：驱动真实 Trainer + 真实数据跑少量迭代，验证 trainer 集成。

验证点：
  - before_train: prepare_qat_model + optimizer + resume + dataloader + EMA
  - train_one_iter: QAT forward + compute_qat_loss + backward + optimizer.step + EMA update
  - save_ckpt: checkpoint 保存
  - after_train: convert + 导出 QDQ-ONNX

用法：
  cd /home/liuyc/workspace/programmers/keypoint/yolox_ks
  CUDA_VISIBLE_DEVICES=2 /opt/miniconda3/envs/lyc_py38/bin/python tests/test_qat_smoke.py \
      -f exps/example/custom/yolox_kp.py [-c weights/pt/yoloxn_DWFT_pre_784.pth]
"""
import argparse
import os
import sys
import faulthandler
faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from yolox.exp import get_exp
from yolox.core.trainer import Trainer


class Args:
    """Minimal args namespace matching what Trainer accesses."""
    fp16 = False
    qat = True
    batch_size = 4
    cache = False
    occupy = False
    logger = "tensorboard"
    ckpt = None
    resume = False
    start_epoch = None
    experiment_name = "qat_smoke"
    devices = 1


def main():
    parser = argparse.ArgumentParser("QAT smoke test")
    parser.add_argument("-f", "--exp_file", required=True)
    parser.add_argument("-c", "--ckpt", default=None)
    parser.add_argument("--n_iters", type=int, default=3)
    cli = parser.parse_args()

    args = Args()
    args.ckpt = cli.ckpt

    exp = get_exp(cli.exp_file, None)
    exp.max_epoch = 1  # 限制为 1 epoch，冒烟用
    exp.eval_interval = 999  # 冒烟阶段不触发完整 COCO eval
    exp.data_num_workers = 0  # 主进程加载，崩溃时可见 traceback

    print(f"[1/6] Building Trainer with --qat (batch={args.batch_size})...")
    trainer = Trainer(exp, args)
    print("    Trainer built OK")

    print("[2/6] before_train (prepare_qat + optimizer + dataloader + EMA)...")
    trainer.before_train()
    print("    before_train OK")

    # Sanity: model is a GraphModule (prepared)
    from torch.fx import GraphModule
    base = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    assert isinstance(base, GraphModule), f"Expected GraphModule, got {type(base)}"
    print(f"    Model is prepared GraphModule: OK")

    print(f"[3/6] Running {cli.n_iters} training iterations on real data...")
    trainer.epoch = 0
    trainer.before_epoch()
    for it in range(cli.n_iters):
        trainer.iter = it
        trainer.train_one_iter()
        loss_val = trainer.meter["total_loss"].latest if "total_loss" in trainer.meter else float('nan')
        print(f"    iter {it}: total_loss={loss_val:.4f}")
    print("    Training iterations OK")

    print("[4/6] Saving checkpoint...")
    trainer.save_ckpt("smoke_test")
    ckpt_path = os.path.join(trainer.file_name, "smoke_test_ckpt.pth")
    assert os.path.exists(ckpt_path), "Checkpoint not saved!"
    # Verify ckpt has fake-quant params (GraphModule state_dict)
    sd = torch.load(ckpt_path, map_location="cpu")["model"]
    n_fq_keys = sum(1 for k in sd if "activation_post_process" in k)
    print(f"    Checkpoint saved: {ckpt_path} ({len(sd)} keys, {n_fq_keys} fake-quant keys)")
    assert n_fq_keys > 0, "No fake-quant keys in checkpoint!"

    print("[4b/6] get_qat_eval_model (eval wrapper for COCO evaluator)...")
    # Full COCO eval is too slow for a smoke test; verify the eval wrapper is
    # constructed correctly and produces the eval [B, N, C] layout that the
    # COCO evaluator consumes. (Wrapper correctness is fully covered by
    # tests/test_qat_eval_wrapper.py.)
    eval_model = trainer.get_qat_eval_model()
    with torch.no_grad():
        H, W = exp.test_size
        eval_out = eval_model(torch.randn(2, exp.img_channel, H, W).to(trainer.device))
    exp_C = 4 + 1 + exp.num_classes + exp.keypoints * 3
    assert eval_out.dim() == 3 and eval_out.shape[2] == exp_C, \
        f"Eval wrapper output wrong shape: {tuple(eval_out.shape)}"
    assert hasattr(eval_model.head, "decode_outputs"), "wrapper.head missing decode_outputs"
    print(f"    eval wrapper output shape {tuple(eval_out.shape)}: OK")

    print("[5/6] after_train (convert + export QDQ-ONNX)...")
    trainer.after_train()
    qat_onnx = os.path.join(trainer.file_name, "qat_best.onnx")
    assert os.path.exists(qat_onnx), f"QAT ONNX not exported at {qat_onnx}!"
    print(f"    QAT ONNX exported: {qat_onnx}")

    print("[6/6] Verifying QDQ nodes in exported ONNX...")
    import onnx
    m = onnx.load(qat_onnx)
    qdq = sum(1 for n in m.graph.node
              if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    n_out = len(m.graph.output)
    print(f"    QDQ nodes: {qdq}, outputs: {n_out}")
    assert qdq > 0, "No QDQ nodes in exported ONNX!"

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED! Full QAT trainer integration works.")
    print(f"  - before_train (prepare+optimizer+EMA): OK")
    print(f"  - {cli.n_iters} training iters (real data, forward+loss+backward): OK")
    print(f"  - checkpoint save ({n_fq_keys} fake-quant keys): OK")
    print(f"  - after_train convert+export ({qdq} QDQ nodes, {n_out} outputs): OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
