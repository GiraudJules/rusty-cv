#!/usr/bin/env python3

from __future__ import annotations

import json
from typing import Sequence


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms(boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_threshold: float) -> list[int]:
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have the same length, got {len(boxes)} boxes and {len(scores)} scores"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in the inclusive range [0.0, 1.0], got {iou_threshold}")

    order = sorted(range(len(boxes)), key=lambda idx: (-scores[idx], idx))
    keep: list[int] = []

    for idx in order:
        suppressed = False
        for kept_idx in keep:
            if iou(boxes[idx], boxes[kept_idx]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)

    return keep


def demo_payload() -> dict:
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
    ]
    scores = [0.9, 0.8, 0.7]
    return {
        "boxes": boxes,
        "scores": scores,
        "iou_threshold": 0.5,
        "keep": nms(boxes, scores, 0.5),
    }


if __name__ == "__main__":
    print(json.dumps(demo_payload(), indent=2))
