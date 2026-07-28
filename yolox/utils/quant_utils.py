"""
QAT (Quantization-Aware Training) utilities for RKNN deployment.

Uses PyTorch FX graph-mode QAT (torch.ao.quantization.quantize_fx),
which is the same underlying mechanism as RKNN's prepare_qat_fx interface.

Workflow:
  1. prepare_qat_model(): wrap model in traceable wrapper, call prepare_qat_fx
  2. Train with prepared model (fake-quant active, loss computed eagerly)
  3. convert_and_export_qat(): convert_fx -> ONNX with QDQ nodes
  4. RKNN: load_onnx() (auto-detects QDQ nodes) + build(do_quantization=False) + export_rknn
"""

import os
import torch
import torch.nn as nn
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import QConfig, FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from loguru import logger


class TraceableYOLOXForQAT(nn.Module):
    """FX-traceable wrapper for QAT.

    Only performs conv computations (backbone + head convolutions),
    returning raw per-scale prediction tensors WITHOUT sigmoid/concat/decode.
    This matches the RKNN deployment path (--rknpu / RKNN_model_hack=1).

    Shares backbone and head references with the original YOLOX model,
    so training this wrapper updates the same underlying weights.

    Output: tuple of 4*n_scales tensors for n_scales x 4 predictions:
      (reg_s8, obj_s8, cls_s8, lmk_s8,
       reg_s16, obj_s16, cls_s16, lmk_s16,
       reg_s32, obj_s32, cls_s32, lmk_s32)
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        fpn_outs = self.backbone(x)
        outputs = []
        for k in range(len(self.head.stems)):
            feat = self.head.stems[k](fpn_outs[k])

            cls_feat = self.head.cls_convs[k](feat)
            cls_output = self.head.cls_preds[k](cls_feat)

            reg_feat = self.head.reg_convs[k](feat)
            reg_output = self.head.reg_preds[k](reg_feat)
            obj_output = self.head.obj_preds[k](reg_feat)
            lmk_output = self.head.lmk_preds[k](reg_feat)

            outputs.append(reg_output)
            outputs.append(obj_output)
            outputs.append(cls_output)
            outputs.append(lmk_output)

        return tuple(outputs)


class QATEvalWrapper(nn.Module):
    """Evaluator-facing wrapper for a QAT prepared GraphModule.

    The COCO evaluator (yolox/evaluators/coco_evaluator.py) expects:
      - model(imgs) -> a single [B, N, C] tensor (eval layout)
      - model.head.decode_in_inference (attribute)
      - model.head.decode_outputs (method, used as decoder when
        decode_in_inference is False)

    But a QAT prepared GraphModule outputs a tuple of 12 raw per-scale
    tensors (TraceableYOLOXForQAT layout). This wrapper bridges the gap:
      - forward() assembles the 12 raw tensors into the eval [B, N, C]
        layout (obj/cls sigmoid'd, reg/lmk raw), matching YOLOXHead's
        eval branch exactly.
      - exposes a `.head` proxy to the ORIGINAL YOLOXHead so the evaluator
        can read decode_in_inference and call decode_outputs.

    Shares the prepared GraphModule's weights, so evaluating this wrapper
    measures the actual (fake-quantized) training weights.
    """

    def __init__(self, prepared, head):
        super().__init__()
        self.prepared = prepared
        self._qat_head = head  # original YOLOXHead (has decode_outputs / strides)

    @property
    def head(self):
        return self._qat_head

    def forward(self, x):
        raw = self.prepared(x)
        n_scales = len(self._qat_head.strides)
        outputs = []
        hw = []
        for k in range(n_scales):
            reg_output = raw[k * 4]
            obj_output = raw[k * 4 + 1]
            cls_output = raw[k * 4 + 2]
            lmk_output = raw[k * 4 + 3]
            # Eval layout (matches YOLOXHead eval branch): obj/cls sigmoid'd
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), lmk_output], 1
            )
            outputs.append(output)
            hw.append(reg_output.shape[-2:])  # (H, W) per scale
        # The evaluator's decoder (head.decode_outputs) reads self.hw, which is
        # normally set inside head.forward. Since we bypass head.forward here,
        # set it explicitly so decode_outputs works.
        self._qat_head.hw = [tuple(h) for h in hw]
        outputs = torch.cat(
            [o.flatten(start_dim=2) for o in outputs], dim=2
        ).permute(0, 2, 1)
        return outputs


def get_qat_qconfig(backend='qnnpack'):
    """Get QAT qconfig suitable for RKNN (ARM NPU).

    Args:
        backend: 'qnnpack' (per-tensor asymmetric, ARM/RKNN-friendly)
                 or 'fbgemm' (per-channel, x86 reference)
    """
    if backend == 'qnnpack':
        # Matches RKNN NPU's recommended qconfig (SDK V2.3.2 guide §120):
        #   activation: uint8 per-tensor asymmetric (0~255), reduce_range=False
        #   weight:     int8 per-channel affine (-128~127)
        # per_channel_affine (not the default per_channel_symmetric) lets
        # zero_point be non-zero, which RKNN hardware supports.
        # reduce_range=False keeps the full -128~127 range (better accuracy).
        qconfig = QConfig(
            activation=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
            ),
            weight=FakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_affine,
                reduce_range=False,
            ),
        )
    else:
        from torch.ao.quantization import get_default_qat_qconfig
        qconfig = get_default_qat_qconfig(backend)
    return qconfig


def _disable_inplace_act(model):
    """Disable inplace on all activation modules (FX QAT compatibility)."""
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.SiLU)):
            m.inplace = False


def _warmup_observer_shapes(prepared, example_input):
    """Run one dummy forward so per-channel observers reach their final shape.

    Per-channel observers start with scale/zero_point buffers of shape [1]
    and only resize to [out_channels] on their first forward in train mode.
    ModelEMA is deepcopy'd before any training forward, so it would keep the
    [1]-shaped buffers and crash in ModelEMA.update() with
    "output with shape [1] doesn't match the broadcast shape [C]".
    Running a single forward here (observers enabled) makes every buffer
    reach its final shape before the EMA copy is made.

    BN running stats are saved/restored so random-input statistics don't
    pollute the pretrained running mean/var.
    """
    device = next(prepared.parameters()).device
    bn_stats = {}
    for name, m in prepared.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and getattr(m, "track_running_stats", False):
            bn_stats[name] = (
                m.running_mean.detach().clone(),
                m.running_var.detach().clone(),
                m.num_batches_tracked.detach().clone(),
            )
    with torch.no_grad():
        prepared(example_input.to(device))
    for name, m in prepared.named_modules():
        if name in bn_stats:
            mean, var, nbt = bn_stats[name]
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
            m.num_batches_tracked.copy_(nbt)


def prepare_qat_model(model, exp, backend='qnnpack'):
    """Prepare model for QAT using FX graph-mode quantization.

    Creates a traceable wrapper, runs prepare_qat_fx to insert fake-quant.
    The returned prepared model (GraphModule) shares conv/bn weights with
    the original model, so optimizer updates propagate to both.

    Args:
        model: YOLOX model (from exp.get_model())
        exp: experiment config (for test_size, img_channel)
        backend: 'qnnpack' for RKNN/ARM, 'fbgemm' for x86

    Returns:
        prepared: GraphModule with fake-quant (use for training forward)
    """
    # Disable inplace activations for FX compatibility
    _disable_inplace_act(model)

    wrapper = TraceableYOLOXForQAT(model)
    wrapper.eval()  # eval mode for tracing

    # qnnpack backend: supports depthwise (groups>1) conv, matches ARM/RKNN target
    torch.backends.quantized.engine = "qnnpack"
    qconfig = get_qat_qconfig(backend)
    # Quantize the whole graph, including the prediction heads. The ONNX
    # exporter emits QDQ nodes around the graph outputs regardless, so the
    # exported model is fully quantized either way — and with QAT training,
    # the heads' observers adapt to represent negative logits correctly.
    qconfig_dict = {"": qconfig}

    example_input = torch.randn(
        1, exp.img_channel, exp.test_size[0], exp.test_size[1]
    )

    logger.info("Preparing QAT model (FX graph-mode, backend={})...".format(backend))
    prepared = prepare_qat_fx(wrapper, qconfig_dict, example_inputs=example_input)

    n_fq = sum(1 for m in prepared.modules() if isinstance(m, FakeQuantize))
    logger.info("QAT prepared: {} FakeQuantize modules inserted".format(n_fq))

    prepared.train()  # back to train mode for QAT training

    # Grow per-channel observer buffers ([1] -> [out_channels]) before any
    # EMA deepcopy — otherwise ModelEMA.update() hits a shape mismatch.
    _warmup_observer_shapes(prepared, example_input)

    return prepared


def compute_qat_loss(head, raw_outputs, imgs, targets, seg_targets=None):
    """Compute training loss from raw per-scale outputs (for QAT training).

    Replicates the head's training-branch loss computation, but takes
    raw per-scale outputs from TraceableYOLOXForQAT instead of computing
    them internally. The conv part goes through fake-quant (GraphModule),
    decode + loss are eager.

    Args:
        head: YOLOXHead instance (shared with original model)
        raw_outputs: tuple of 4*n_scales tensors from prepared model
        imgs: input images tensor [B, C, H, W]
        targets: ground truth labels [B, max_labels, ...]
        seg_targets: segmentation targets (None if segcls=0)

    Returns:
        dict with 'total_loss' and individual loss components
    """
    n_scales = len(head.strides)
    outputs_per_scale = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []
    origin_preds = []

    for k in range(n_scales):
        reg_output = raw_outputs[k * 4]
        obj_output = raw_outputs[k * 4 + 1]
        cls_output = raw_outputs[k * 4 + 2]
        lmk_output = raw_outputs[k * 4 + 3]

        stride = head.strides[k]

        # Same as training branch: cat and decode
        if head.keypoints > 0:
            output = torch.cat([reg_output, obj_output, cls_output, lmk_output], 1)
        elif head.segcls > 0:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
        else:
            output = torch.cat([reg_output, obj_output, cls_output], 1)

        output, grid = head.get_output_and_grid(
            output, k, stride, imgs.type()  # type string includes device (e.g. 'torch.cuda.FloatTensor')
        )
        x_shifts.append(grid[:, :, 0])
        y_shifts.append(grid[:, :, 1])
        expanded_strides.append(
            torch.zeros(1, grid.shape[1])
                .fill_(stride)
                .type_as(imgs)
        )

        # L1 origin preds (same as training branch)
        if head.use_l1:
            batch_size = reg_output.shape[0]
            hsize, wsize = reg_output.shape[-2:]
            reg_out = reg_output.view(
                batch_size, head.n_anchors, 4, hsize, wsize
            )
            reg_out = reg_out.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
            origin_preds.append(reg_out.clone())

        outputs_per_scale.append(output)

    # Call head's get_losses (same as original training)
    (loss, iou_loss, conf_loss, cls_loss, l1_loss,
     lmk_loss, seg_loss, num_fg) = head.get_losses(
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        targets,
        torch.cat(outputs_per_scale, 1),
        origin_preds,
        dtype=imgs.dtype,
        seg_proto=None,
        semantic_pred=None,
        seg_labels=seg_targets,
    )

    outputs = {
        "total_loss": loss,
        "iou_loss": iou_loss,
        "l1_loss": l1_loss,
        "conf_loss": conf_loss,
        "cls_loss": cls_loss,
        "num_fg": num_fg,
    }
    if head.keypoints > 0:
        outputs['kp_loss'] = lmk_loss
    if head.segcls > 0:
        outputs['seg_loss'] = seg_loss

    return outputs


def convert_and_export_qat(prepared, exp, output_name, opset=19, no_onnxsim=True):
    """Convert prepared QAT model and export ONNX with QDQ nodes.

    Args:
        prepared: GraphModule from prepare_qat_fx (after QAT training)
        exp: experiment config
        output_name: output ONNX file path
        opset: ONNX opset (>=13 for QDQ, default 19)
        no_onnxsim: skip onnxsim (recommended for QDQ models)

    Returns:
        output_name: path to exported ONNX
    """
    prepared.eval()
    # Use qnnpack backend: supports depthwise (groups>1) quantized conv and
    # matches the ARM/RKNN target. The default fbgemm/cudnn backend fails on
    # depthwise conv ("cudnn conv2d limited to groups=1").
    # qnnpack requires CPU tensors for convert_fx.
    torch.backends.quantized.engine = "qnnpack"
    prepared = prepared.cpu()
    logger.info("Converting QAT model (inserting QDQ nodes, engine=qnnpack, CPU)...")
    quantized = convert_fx(prepared)

    dummy_input = torch.randn(
        1, exp.img_channel, exp.test_size[0], exp.test_size[1]
    )

    logger.info("Exporting QAT ONNX (opset={})...".format(opset))
    torch.onnx.export(
        quantized,
        dummy_input,
        output_name,
        input_names=["images"],
        output_names=["output"],
        opset_version=opset,
    )
    logger.info("QAT ONNX exported to {}".format(output_name))

    if not no_onnxsim:
        try:
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(output_name)
            model_simp, check = simplify(onnx_model)
            if check:
                onnx.save(model_simp, output_name)
                logger.info("Simplified QAT ONNX saved")
            else:
                logger.warning("onnxsim failed, keeping unsimplified model")
        except ImportError:
            logger.warning("onnx/onnxsim not installed, skipping simplification")

    # Verify QDQ nodes
    try:
        import onnx
        onnx_model = onnx.load(output_name)
        qdq_count = sum(
            1 for node in onnx_model.graph.node
            if node.op_type in ('QuantizeLinear', 'DequantizeLinear')
        )
        n_outputs = len(onnx_model.graph.output)
        logger.info(
            "QDQ verification: {} QuantizeLinear/DequantizeLinear nodes, "
            "{} outputs".format(qdq_count, n_outputs)
        )
        if qdq_count == 0:
            logger.warning(
                "No QDQ nodes! ONNX may not be suitable for RKNN QAT."
            )
    except ImportError:
        logger.info("Install 'onnx' to verify QDQ nodes in exported model")

    logger.info("=" * 60)
    logger.info("Next: convert to RKNN (in rknn2.3.2 env):")
    logger.info("  Use onnx2rknn2_qat.py (load_onnx auto-detects QDQ nodes,")
    logger.info("  build with do_quantization=False)")
    logger.info("=" * 60)

    return output_name
