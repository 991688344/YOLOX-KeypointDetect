#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        ckpt_key = key_model
        if ckpt_key not in ckpt:
            # QAT GraphModule ckpts unfold BaseConv, so BN keys gain an
            # extra '.conv' segment: head.stems.0.bn.* -> head.stems.0.conv.bn.*
            alt_key = key_model.replace('.bn.', '.conv.bn.')
            if alt_key in ckpt and ckpt[alt_key].shape == v.shape:
                ckpt_key = alt_key
            else:
                logger.warning(
                    "{} is not in the ckpt. Please double check and see if this is desired.".format(
                        key_model
                    )
                )
                continue
        v_ckpt = ckpt[ckpt_key]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
        curr_ap = state.get("curr_ap")
        curr_kp_ap = state.get("curr_kp_ap")
        if curr_ap is not None and curr_kp_ap is not None:
            logger.info("New best AP50_95: {:.2f}, KP_AP50_95: {:.2f}, saving best weights: {}.".format(curr_ap * 100, curr_kp_ap * 100, best_filename))
        else:
            # QAT 等无 COCO 评估指标的场景：无 AP 值，仅保存 best 权重
            logger.info("Saving best weights: {}.".format(best_filename))
