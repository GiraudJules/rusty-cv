#!/usr/bin/env python3

from __future__ import annotations

import sys
import time

import numpy as np
import rusty_cv

from nms_ref import nms as python_nms


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


def benchmark_python(iterations: int, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> tuple[float, int]:
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        keep = python_nms(boxes_list, scores_list, iou_threshold)
        checksum += sum(keep)
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def benchmark_rust(iterations: int, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> tuple[float, int]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        keep = rusty_cv.nms(boxes, scores, iou_threshold=iou_threshold)
        checksum += sum(keep)
    elapsed = time.perf_counter() - start
    return elapsed, checksum


def main() -> None:
    box_count = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    iou_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    boxes, scores = synthetic_boxes(box_count)

    rust_keep = rusty_cv.nms(boxes, scores, iou_threshold=iou_threshold)
    python_keep = python_nms(boxes.tolist(), scores.tolist(), iou_threshold)
    if rust_keep != python_keep:
        raise SystemExit(
            f"nms mismatch between Rust and Python reference\nrust={rust_keep}\npython={python_keep}"
        )

    rust_elapsed, rust_checksum = benchmark_rust(iterations, boxes, scores, iou_threshold)
    python_elapsed, python_checksum = benchmark_python(iterations, boxes, scores, iou_threshold)

    print(f"boxes={box_count}")
    print(f"iterations={iterations}")
    print(f"iou_threshold={iou_threshold}")
    print(f"rust_elapsed_s={rust_elapsed:.6f}")
    print(f"rust_us_per_iter={(rust_elapsed * 1e6) / iterations:.2f}")
    print(f"rust_checksum={rust_checksum}")
    print(f"python_elapsed_s={python_elapsed:.6f}")
    print(f"python_us_per_iter={(python_elapsed * 1e6) / iterations:.2f}")
    print(f"python_checksum={python_checksum}")
    print(f"speedup={(python_elapsed / rust_elapsed):.2f}x")


if __name__ == "__main__":
    main()
