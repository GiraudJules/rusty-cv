#!/usr/bin/env python3

from __future__ import annotations

import json
import math
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


def _sort_indices(indices: list[int], scores: Sequence[float]) -> list[int]:
    return sorted(indices, key=lambda idx: (-scores[idx], idx))


def _nms_from_sorted_indices(
    boxes: Sequence[Sequence[float]],
    sorted_indices: Sequence[int],
    iou_threshold: float,
) -> list[int]:
    keep: list[int] = []

    for idx in sorted_indices:
        suppressed = False
        for kept_idx in keep:
            if iou(boxes[idx], boxes[kept_idx]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)

    return keep


def nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
    *,
    score_threshold: float = float("-inf"),
    pre_nms_top_k: int | None = None,
    max_detections: int | None = None,
) -> list[int]:
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have the same length, got {len(boxes)} boxes and {len(scores)} scores"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in the inclusive range [0.0, 1.0], got {iou_threshold}")

    candidates = [idx for idx, score in enumerate(scores) if score >= score_threshold]
    ordered = _sort_indices(candidates, scores)
    if pre_nms_top_k is not None:
        ordered = ordered[:pre_nms_top_k]

    keep = _nms_from_sorted_indices(boxes, ordered, iou_threshold)
    if max_detections is not None:
        keep = keep[:max_detections]
    return keep


def batched_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    class_ids: Sequence[int],
    iou_threshold: float,
    *,
    score_threshold: float = float("-inf"),
    pre_nms_top_k: int | None = None,
    max_detections: int | None = None,
) -> dict[str, list[float] | list[int]]:
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have the same length, got {len(boxes)} boxes and {len(scores)} scores"
        )
    if len(boxes) != len(class_ids):
        raise ValueError(
            f"boxes and class_ids must have the same length, got {len(boxes)} boxes and {len(class_ids)} class_ids"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in the inclusive range [0.0, 1.0], got {iou_threshold}")

    detections: list[tuple[int, int, float]] = []
    for class_id in sorted(set(class_ids)):
        candidates = [idx for idx, score in enumerate(scores) if class_ids[idx] == class_id and score >= score_threshold]
        ordered = _sort_indices(candidates, scores)
        if pre_nms_top_k is not None:
            ordered = ordered[:pre_nms_top_k]
        keep = _nms_from_sorted_indices(boxes, ordered, iou_threshold)
        for idx in keep:
            detections.append((idx, class_id, scores[idx]))

    detections.sort(key=lambda detection: (-detection[2], detection[0], detection[1]))
    if max_detections is not None:
        detections = detections[:max_detections]

    return {
        "indices": [detection[0] for detection in detections],
        "class_ids": [detection[1] for detection in detections],
        "scores": [detection[2] for detection in detections],
    }


def multiclass_nms(
    boxes: Sequence[Sequence[float]],
    class_scores: Sequence[Sequence[float]],
    iou_threshold: float,
    *,
    score_threshold: float = float("-inf"),
    pre_nms_top_k: int | None = None,
    max_detections: int | None = None,
) -> dict[str, list[float] | list[int]]:
    if not class_scores:
        return {"indices": [], "class_ids": [], "scores": []}
    if len(class_scores) != len(boxes):
        raise ValueError(
            f"class_scores must have one row per box, got {len(boxes)} boxes and {len(class_scores)} score rows"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in the inclusive range [0.0, 1.0], got {iou_threshold}")

    num_classes = len(class_scores[0])
    detections: list[tuple[int, int, float]] = []

    for class_id in range(num_classes):
        candidates = [
            idx
            for idx, scores in enumerate(class_scores)
            if scores[class_id] >= score_threshold
        ]
        ordered = sorted(candidates, key=lambda idx: (-class_scores[idx][class_id], idx))
        if pre_nms_top_k is not None:
            ordered = ordered[:pre_nms_top_k]
        keep = _nms_from_sorted_indices(boxes, ordered, iou_threshold)
        for idx in keep:
            detections.append((idx, class_id, class_scores[idx][class_id]))

    detections.sort(key=lambda detection: (-detection[2], detection[0], detection[1]))
    if max_detections is not None:
        detections = detections[:max_detections]

    return {
        "indices": [detection[0] for detection in detections],
        "class_ids": [detection[1] for detection in detections],
        "scores": [detection[2] for detection in detections],
    }


def soft_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    *,
    method: str = "linear",
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    sigma: float = 0.5,
    pre_nms_top_k: int | None = None,
    max_detections: int | None = None,
) -> dict[str, list[float] | list[int]]:
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have the same length, got {len(boxes)} boxes and {len(scores)} scores"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in the inclusive range [0.0, 1.0], got {iou_threshold}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be greater than zero, got {sigma}")
    if method not in {"linear", "gaussian"}:
        raise ValueError(f"unsupported soft_nms method {method!r}")

    ordered = _sort_indices(list(range(len(boxes))), scores)
    if pre_nms_top_k is not None:
        ordered = ordered[:pre_nms_top_k]

    candidates = [
        {"index": idx, "score": float(scores[idx])}
        for idx in ordered
    ]
    detections: list[tuple[int, int, float]] = []

    while candidates:
        best_pos = max(
            range(len(candidates)),
            key=lambda pos: (candidates[pos]["score"], -candidates[pos]["index"]),
        )
        best = candidates.pop(best_pos)
        if best["score"] < score_threshold:
            break

        detections.append((best["index"], 0, best["score"]))
        if max_detections is not None and len(detections) >= max_detections:
            break

        best_box = boxes[best["index"]]
        remaining: list[dict[str, float | int]] = []
        for candidate in candidates:
            overlap = iou(best_box, boxes[candidate["index"]])
            if method == "linear":
                weight = 1.0 - overlap if overlap > iou_threshold else 1.0
            else:
                weight = math.exp(-(overlap * overlap) / sigma)
            candidate["score"] *= weight
            if candidate["score"] >= score_threshold:
                remaining.append(candidate)
        candidates = remaining

    return {
        "indices": [detection[0] for detection in detections],
        "class_ids": [detection[1] for detection in detections],
        "scores": [detection[2] for detection in detections],
    }


def batched_soft_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    class_ids: Sequence[int],
    *,
    method: str = "linear",
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    sigma: float = 0.5,
    pre_nms_top_k: int | None = None,
    max_detections: int | None = None,
) -> dict[str, list[float] | list[int]]:
    if len(boxes) != len(scores):
        raise ValueError(
            f"boxes and scores must have the same length, got {len(boxes)} boxes and {len(scores)} scores"
        )
    if len(boxes) != len(class_ids):
        raise ValueError(
            f"boxes and class_ids must have the same length, got {len(boxes)} boxes and {len(class_ids)} class_ids"
        )

    detections: list[tuple[int, int, float]] = []
    for class_id in sorted(set(class_ids)):
        candidates = [idx for idx, score in enumerate(scores) if class_ids[idx] == class_id and score >= score_threshold]
        ordered = _sort_indices(candidates, scores)
        if pre_nms_top_k is not None:
            ordered = ordered[:pre_nms_top_k]
        class_result = soft_nms(
            [boxes[idx] for idx in ordered],
            [scores[idx] for idx in ordered],
            method=method,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            sigma=sigma,
        )
        for relative_idx, score in zip(class_result["indices"], class_result["scores"]):
            detections.append((ordered[relative_idx], class_id, score))

    detections.sort(key=lambda detection: (-detection[2], detection[0], detection[1]))
    if max_detections is not None:
        detections = detections[:max_detections]

    return {
        "indices": [detection[0] for detection in detections],
        "class_ids": [detection[1] for detection in detections],
        "scores": [detection[2] for detection in detections],
    }


def multiclass_soft_nms(
    boxes: Sequence[Sequence[float]],
    class_scores: Sequence[Sequence[float]],
    *,
    method: str = "linear",
    iou_threshold: float = 0.5,
    score_threshold: float = float("-inf"),
    sigma: float = 0.5,
    pre_nms_top_k: int | None = None,
    max_detections: int | None = None,
) -> dict[str, list[float] | list[int]]:
    if not class_scores:
        return {"indices": [], "class_ids": [], "scores": []}
    if len(class_scores) != len(boxes):
        raise ValueError(
            f"class_scores must have one row per box, got {len(boxes)} boxes and {len(class_scores)} score rows"
        )

    num_classes = len(class_scores[0])
    detections: list[tuple[int, int, float]] = []
    for class_id in range(num_classes):
        candidates = [
            idx
            for idx, scores_row in enumerate(class_scores)
            if scores_row[class_id] >= score_threshold
        ]
        ordered = sorted(candidates, key=lambda idx: (-class_scores[idx][class_id], idx))
        if pre_nms_top_k is not None:
            ordered = ordered[:pre_nms_top_k]
        class_result = soft_nms(
            [boxes[idx] for idx in ordered],
            [class_scores[idx][class_id] for idx in ordered],
            method=method,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            sigma=sigma,
        )
        for relative_idx, score in zip(class_result["indices"], class_result["scores"]):
            detections.append((ordered[relative_idx], class_id, score))

    detections.sort(key=lambda detection: (-detection[2], detection[0], detection[1]))
    if max_detections is not None:
        detections = detections[:max_detections]

    return {
        "indices": [detection[0] for detection in detections],
        "class_ids": [detection[1] for detection in detections],
        "scores": [detection[2] for detection in detections],
    }


def demo_payload() -> dict:
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
    ]
    scores = [0.9, 0.8, 0.7]
    class_ids = [0, 0, 1]
    class_scores = [
        [0.9, 0.1],
        [0.8, 0.75],
        [0.1, 0.7],
    ]
    return {
        "nms": nms(boxes, scores, 0.5),
        "batched_nms": batched_nms(boxes, scores, class_ids, 0.5),
        "multiclass_nms": multiclass_nms(boxes, class_scores, 0.5),
        "soft_nms": soft_nms(boxes, scores),
        "batched_soft_nms": batched_soft_nms(boxes, scores, class_ids),
        "multiclass_soft_nms": multiclass_soft_nms(boxes, class_scores),
    }


if __name__ == "__main__":
    print(json.dumps(demo_payload(), indent=2))
