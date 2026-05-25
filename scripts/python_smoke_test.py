#!/usr/bin/env python3

from __future__ import annotations

import base64

import numpy as np
import rusty_cv


PNG_1X1_RED = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def main() -> None:
    info = rusty_cv.compute_letterbox(1920, 1080, 640, 640)
    assert info["resized_width"] == 640
    assert info["resized_height"] == 360
    assert info["padding"]["top"] == 140
    assert info["padding"]["bottom"] == 140

    resized = rusty_cv.resize_image(PNG_1X1_RED, 4, 2, filter="nearest", output_format="png")
    letterboxed = rusty_cv.letterbox_image(
        PNG_1X1_RED,
        4,
        4,
        fill=(114, 114, 114),
        filter="nearest",
        output_format="png",
    )

    assert resized.startswith(b"\x89PNG\r\n\x1a\n")
    assert letterboxed.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(resized) > 0
    assert len(letterboxed) > 0
    assert rusty_cv.crop_image(PNG_1X1_RED, 0, 0, 1, 1, output_format="png").startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert rusty_cv.center_crop_image(PNG_1X1_RED, 1, 1, output_format="png").startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    array = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    resized_array = rusty_cv.resize_image_numpy(array, 4, 3, filter="nearest")
    letterboxed_array, letterboxed_info = rusty_cv.letterbox_image_numpy(
        array,
        4,
        4,
        fill=(114, 114, 114),
        filter="nearest",
    )

    assert isinstance(resized_array, np.ndarray)
    assert resized_array.dtype == np.uint8
    assert resized_array.shape == (3, 4, 3)

    assert isinstance(letterboxed_array, np.ndarray)
    assert letterboxed_array.dtype == np.uint8
    assert letterboxed_array.shape == (4, 4, 3)
    assert letterboxed_info["resized_width"] == 4
    assert letterboxed_info["resized_height"] == 4

    cropped_array, cropped_info = rusty_cv.crop_image_numpy(array, 1, 0, 1, 2)
    assert cropped_array.shape == (2, 1, 3)
    assert cropped_info["x"] == 1
    assert cropped_info["y"] == 0
    assert cropped_array[0, 0].tolist() == [0, 255, 0]

    centered_array, centered_info = rusty_cv.center_crop_image_numpy(array, 1, 2)
    assert centered_array.shape == (2, 1, 3)
    assert centered_info["x"] == 0
    assert centered_info["y"] == 0

    normalized = rusty_cv.normalize_image_numpy(
        array,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        scale_to_unit=True,
    )
    assert normalized.dtype == np.float32
    assert normalized.shape == (2, 2, 3)
    assert np.isclose(normalized[0, 0, 0], 1.0)
    assert np.isclose(normalized[0, 0, 1], 0.0)

    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = rusty_cv.nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]
    assert np.isclose(rusty_cv.iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)), 25.0 / 175.0)

    print("python smoke test: ok")


if __name__ == "__main__":
    main()
