#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

BBOX_CLASSES = (
    "Person",
    "Seatbelt",
    "Face",
    "Smoking",
    "Phone"
)
KEYPOINT_CLASSES = {
    "Person": [
        "head",
        "neck",
        "left_shoulder",
        "right_shoulder"
    ],
    "Seatbelt": [
        "top",
        "tail"
    ],
}
PLATE_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
                    "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
                    "V", "W", "X", "Y", "Z"]
