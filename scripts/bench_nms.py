#!/usr/bin/env python3

from __future__ import annotations

import sys
import time

import numpy as np
import rusty_cv

from nms_ref import batched_nms as python_batched_nms
from nms_ref import batched_soft_nms as python_batched_soft_nms
from nms_ref import multiclass_nms as python_multiclass_nms
from nms_ref import multiclass_soft_nms as python_multiclass_soft_nms
from nms_ref import nms as python_nms
from nms_ref import soft_nms as python_soft_nms


def synthetic_boxes(count: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.zeros((count, 4), dtype=np.float32)
    scores = np.zeros((count,), dtype=np.float32)

    for i in range(count):
        x1 = float((i * 13) % 320)
        y1 = float((i * 17) % 240)
        width = 20.0 + float(i % 25)
        height = 18.0 + float(i % 21)
        boxes[i] = [x1, y1, x1 + width, y1 + height]
        scores[i] = 1.0 - (i / max(count, 1)) * 0.5

    return boxes, scores


def synthetic_class_ids(count: int, num_classes: int) -> np.ndarray:
    class_ids = np.zeros((count,), dtype=np.int64)
    for i in range(count):
        class_ids[i] = i % max(num_classes, 1)
    return class_ids


def synthetic_class_scores(scores: np.ndarray, num_classes: int) -> np.ndarray:
    class_scores = np.zeros((scores.shape[0], num_classes), dtype=np.float32)
    for box_index, base_score in enumerate(scores):
        for class_id in range(num_classes):
            decay = 0.07 * class_id + 0.02 * ((box_index + class_id) % 3)
            class_scores[box_index, class_id] = max(0.0, float(base_score) - decay)
    return class_scores


def benchmark_single_python(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        keep = python_nms(boxes_list, scores_list, iou_threshold)
        checksum += sum(keep)
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_single_rust(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        keep = rusty_cv.nms(boxes, scores, iou_threshold=iou_threshold)
        checksum += sum(keep)
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_batched_python(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    class_ids_list = class_ids.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = python_batched_nms(boxes_list, scores_list, class_ids_list, iou_threshold)
        checksum += sum(result["indices"]) + sum(result["class_ids"])
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_batched_rust(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = rusty_cv.batched_nms(boxes, scores, class_ids, iou_threshold=iou_threshold)
        checksum += int(result["indices"].sum()) + int(result["class_ids"].sum())
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_multiclass_python(
    iterations: int,
    boxes: np.ndarray,
    class_scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    class_scores_list = class_scores.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = python_multiclass_nms(
            boxes_list,
            class_scores_list,
            iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += sum(result["indices"]) + sum(result["class_ids"])
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_multiclass_rust(
    iterations: int,
    boxes: np.ndarray,
    class_scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = rusty_cv.multiclass_nms(
            boxes,
            class_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += int(result["indices"].sum()) + int(result["class_ids"].sum())
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_soft_python(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = python_soft_nms(
            boxes_list,
            scores_list,
            method="linear",
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += sum(result["indices"])
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_soft_rust(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = rusty_cv.soft_nms(
            boxes,
            scores,
            method="linear",
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += int(result["indices"].sum())
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_batched_soft_python(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    class_ids_list = class_ids.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = python_batched_soft_nms(
            boxes_list,
            scores_list,
            class_ids_list,
            method="linear",
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += sum(result["indices"]) + sum(result["class_ids"])
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_batched_soft_rust(
    iterations: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = rusty_cv.batched_soft_nms(
            boxes,
            scores,
            class_ids,
            method="linear",
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += int(result["indices"].sum()) + int(result["class_ids"].sum())
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_multiclass_soft_python(
    iterations: int,
    boxes: np.ndarray,
    class_scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    class_scores_list = class_scores.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = python_multiclass_soft_nms(
            boxes_list,
            class_scores_list,
            method="linear",
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += sum(result["indices"]) + sum(result["class_ids"])
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_multiclass_soft_rust(
    iterations: int,
    boxes: np.ndarray,
    class_scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        result = rusty_cv.multiclass_soft_nms(
            boxes,
            class_scores,
            method="linear",
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        checksum += int(result["indices"].sum()) + int(result["class_ids"].sum())
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def print_comparison(
    label: str,
    iterations: int,
    rust_elapsed: float,
    rust_checksum: int,
    python_elapsed: float,
    python_checksum: int,
) -> None:
    print(f"[{label}]")
    print(f"rust_elapsed_s={rust_elapsed:.6f}")
    print(f"rust_us_per_iter={(rust_elapsed * 1e6) / iterations:.2f}")
    print(f"rust_checksum={rust_checksum}")
    print(f"python_elapsed_s={python_elapsed:.6f}")
    print(f"python_us_per_iter={(python_elapsed * 1e6) / iterations:.2f}")
    print(f"python_checksum={python_checksum}")
    print(f"speedup={(python_elapsed / rust_elapsed):.2f}x")


def main() -> None:
    box_count = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    iou_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    num_classes = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    score_threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25

    boxes, scores = synthetic_boxes(box_count)
    class_ids = synthetic_class_ids(box_count, num_classes)
    class_scores = synthetic_class_scores(scores, num_classes)

    rust_keep = rusty_cv.nms(boxes, scores, iou_threshold=iou_threshold)
    python_keep = python_nms(boxes.tolist(), scores.tolist(), iou_threshold)
    if rust_keep != python_keep:
        raise SystemExit(
            f"nms mismatch between Rust and Python reference\nrust={rust_keep}\npython={python_keep}"
        )

    rust_batched = rusty_cv.batched_nms(boxes, scores, class_ids, iou_threshold=iou_threshold)
    python_batched = python_batched_nms(boxes.tolist(), scores.tolist(), class_ids.tolist(), iou_threshold)
    if rust_batched["indices"].tolist() != python_batched["indices"] or rust_batched["class_ids"].tolist() != python_batched["class_ids"]:
        raise SystemExit(
            "batched_nms mismatch between Rust and Python reference\n"
            f"rust={rust_batched}\npython={python_batched}"
        )

    rust_multiclass = rusty_cv.multiclass_nms(
        boxes,
        class_scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    python_multiclass = python_multiclass_nms(
        boxes.tolist(),
        class_scores.tolist(),
        iou_threshold,
        score_threshold=score_threshold,
    )
    if (
        rust_multiclass["indices"].tolist() != python_multiclass["indices"]
        or rust_multiclass["class_ids"].tolist() != python_multiclass["class_ids"]
    ):
        raise SystemExit(
            "multiclass_nms mismatch between Rust and Python reference\n"
            f"rust={rust_multiclass}\npython={python_multiclass}"
        )

    rust_soft = rusty_cv.soft_nms(
        boxes,
        scores,
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    python_soft = python_soft_nms(
        boxes.tolist(),
        scores.tolist(),
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    if rust_soft["indices"].tolist() != python_soft["indices"]:
        raise SystemExit(
            "soft_nms mismatch between Rust and Python reference\n"
            f"rust={rust_soft}\npython={python_soft}"
        )

    rust_batched_soft = rusty_cv.batched_soft_nms(
        boxes,
        scores,
        class_ids,
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    python_batched_soft = python_batched_soft_nms(
        boxes.tolist(),
        scores.tolist(),
        class_ids.tolist(),
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    if (
        rust_batched_soft["indices"].tolist() != python_batched_soft["indices"]
        or rust_batched_soft["class_ids"].tolist() != python_batched_soft["class_ids"]
    ):
        raise SystemExit(
            "batched_soft_nms mismatch between Rust and Python reference\n"
            f"rust={rust_batched_soft}\npython={python_batched_soft}"
        )

    rust_multiclass_soft = rusty_cv.multiclass_soft_nms(
        boxes,
        class_scores,
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    python_multiclass_soft = python_multiclass_soft_nms(
        boxes.tolist(),
        class_scores.tolist(),
        method="linear",
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    if (
        rust_multiclass_soft["indices"].tolist() != python_multiclass_soft["indices"]
        or rust_multiclass_soft["class_ids"].tolist() != python_multiclass_soft["class_ids"]
    ):
        raise SystemExit(
            "multiclass_soft_nms mismatch between Rust and Python reference\n"
            f"rust={rust_multiclass_soft}\npython={python_multiclass_soft}"
        )

    rust_elapsed, rust_checksum = benchmark_single_rust(iterations, boxes, scores, iou_threshold)
    python_elapsed, python_checksum = benchmark_single_python(iterations, boxes, scores, iou_threshold)
    rust_batched_elapsed, rust_batched_checksum = benchmark_batched_rust(
        iterations, boxes, scores, class_ids, iou_threshold
    )
    python_batched_elapsed, python_batched_checksum = benchmark_batched_python(
        iterations, boxes, scores, class_ids, iou_threshold
    )
    rust_multi_elapsed, rust_multi_checksum = benchmark_multiclass_rust(
        iterations, boxes, class_scores, iou_threshold, score_threshold
    )
    python_multi_elapsed, python_multi_checksum = benchmark_multiclass_python(
        iterations, boxes, class_scores, iou_threshold, score_threshold
    )
    rust_soft_elapsed, rust_soft_checksum = benchmark_soft_rust(
        iterations, boxes, scores, iou_threshold, score_threshold
    )
    python_soft_elapsed, python_soft_checksum = benchmark_soft_python(
        iterations, boxes, scores, iou_threshold, score_threshold
    )
    rust_batched_soft_elapsed, rust_batched_soft_checksum = benchmark_batched_soft_rust(
        iterations, boxes, scores, class_ids, iou_threshold, score_threshold
    )
    python_batched_soft_elapsed, python_batched_soft_checksum = benchmark_batched_soft_python(
        iterations, boxes, scores, class_ids, iou_threshold, score_threshold
    )
    rust_multiclass_soft_elapsed, rust_multiclass_soft_checksum = benchmark_multiclass_soft_rust(
        iterations, boxes, class_scores, iou_threshold, score_threshold
    )
    (
        python_multiclass_soft_elapsed,
        python_multiclass_soft_checksum,
    ) = benchmark_multiclass_soft_python(
        iterations, boxes, class_scores, iou_threshold, score_threshold
    )

    print(f"boxes={box_count}")
    print(f"iterations={iterations}")
    print(f"iou_threshold={iou_threshold}")
    print(f"num_classes={num_classes}")
    print(f"score_threshold={score_threshold}")
    print_comparison(
        "single",
        iterations,
        rust_elapsed,
        rust_checksum,
        python_elapsed,
        python_checksum,
    )
    print_comparison(
        "batched",
        iterations,
        rust_batched_elapsed,
        rust_batched_checksum,
        python_batched_elapsed,
        python_batched_checksum,
    )
    print_comparison(
        "multiclass",
        iterations,
        rust_multi_elapsed,
        rust_multi_checksum,
        python_multi_elapsed,
        python_multi_checksum,
    )
    print_comparison(
        "soft",
        iterations,
        rust_soft_elapsed,
        rust_soft_checksum,
        python_soft_elapsed,
        python_soft_checksum,
    )
    print_comparison(
        "soft_batched",
        iterations,
        rust_batched_soft_elapsed,
        rust_batched_soft_checksum,
        python_batched_soft_elapsed,
        python_batched_soft_checksum,
    )
    print_comparison(
        "soft_multiclass",
        iterations,
        rust_multiclass_soft_elapsed,
        rust_multiclass_soft_checksum,
        python_multiclass_soft_elapsed,
        python_multiclass_soft_checksum,
    )


if __name__ == "__main__":
    main()
