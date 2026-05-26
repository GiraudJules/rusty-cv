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

    preprocessed, preprocess_info = rusty_cv.preprocess_image_numpy(
        array,
        4,
        4,
        mode="letterbox",
        fill=(114, 114, 114),
        filter="nearest",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        scale_to_unit=True,
        layout="chw",
    )
    assert preprocessed.dtype == np.float32
    assert preprocessed.shape == (3, 4, 4)
    assert preprocess_info["mode"] == "letterbox"
    assert preprocess_info["layout"] == "chw"
    assert preprocess_info["geometry"]["resized_width"] == 4
    assert preprocess_info["geometry"]["padding"]["top"] == 0

    resized_preprocessed, resized_preprocess_info = rusty_cv.preprocess_image_numpy(
        array,
        3,
        2,
        mode="resize",
        filter="nearest",
        mean=(0.0, 0.0, 0.0),
        std=(255.0, 255.0, 255.0),
        scale_to_unit=False,
        layout="hwc",
    )
    assert resized_preprocessed.shape == (2, 3, 3)
    assert resized_preprocess_info["mode"] == "resize"
    assert resized_preprocess_info["layout"] == "hwc"
    assert np.isclose(resized_preprocessed[0, 0, 0], 1.0)

    hwc_float = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    chw_float = rusty_cv.hwc_to_chw_numpy(hwc_float)
    assert chw_float.dtype == np.float32
    assert chw_float.shape == (3, 2, 4)
    assert np.allclose(chw_float, np.transpose(hwc_float, (2, 0, 1)))
    assert np.allclose(rusty_cv.chw_to_hwc_numpy(chw_float), hwc_float)

    bgr_array = rusty_cv.rgb_to_bgr_numpy(array)
    assert bgr_array.dtype == np.uint8
    assert bgr_array.shape == array.shape
    assert bgr_array[0, 0].tolist() == [0, 0, 255]

    nhwc_batch = np.stack([hwc_float, hwc_float + 100.0], axis=0)
    nchw_batch = rusty_cv.nhwc_to_nchw_numpy(nhwc_batch)
    assert nchw_batch.dtype == np.float32
    assert nchw_batch.shape == (2, 3, 2, 4)
    assert np.allclose(nchw_batch, np.transpose(nhwc_batch, (0, 3, 1, 2)))
    assert np.allclose(rusty_cv.nchw_to_nhwc_numpy(nchw_batch), nhwc_batch)

    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    xywh_boxes = rusty_cv.xyxy_to_xywh_numpy(boxes)
    assert np.allclose(xywh_boxes[0], np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32))
    assert np.allclose(rusty_cv.xywh_to_xyxy_numpy(xywh_boxes), boxes)

    clipped_boxes = rusty_cv.clip_boxes_numpy(
        np.array([[-5.0, 3.0, 12.0, 30.0]], dtype=np.float32),
        10,
        20,
    )
    assert np.allclose(clipped_boxes[0], np.array([0.0, 3.0, 10.0, 20.0], dtype=np.float32))

    score_values = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    filtered_by_score = rusty_cv.filter_boxes_by_score_numpy(boxes, score_values, 0.75)
    assert filtered_by_score["indices"].tolist() == [0, 1]
    assert np.allclose(
        filtered_by_score["boxes"],
        np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
            ],
            dtype=np.float32,
        ),
    )

    filtered_by_area = rusty_cv.filter_boxes_by_area_numpy(boxes, min_area=90.0, max_area=110.0)
    assert filtered_by_area["indices"].tolist() == [0, 1, 2]

    filtered_by_min_size = rusty_cv.filter_boxes_by_min_size_numpy(boxes, 10.0, 10.0)
    assert filtered_by_min_size["indices"].tolist() == [0, 1, 2]

    clipped_and_filtered = rusty_cv.clip_and_filter_boxes_numpy(
        np.array(
            [
                [-5.0, 0.0, 6.0, 9.0],
                [4.0, 4.0, 5.0, 5.0],
                [9.0, 9.0, 15.0, 15.0],
            ],
            dtype=np.float32,
        ),
        10,
        10,
        min_width=2.0,
        min_height=2.0,
    )
    assert clipped_and_filtered["indices"].tolist() == [0]
    assert np.allclose(
        clipped_and_filtered["boxes"],
        np.array([[-0.0, 0.0, 6.0, 9.0]], dtype=np.float32),
    )

    resized_boxes = rusty_cv.resize_boxes_numpy(
        np.array([[10.0, 20.0, 40.0, 60.0]], dtype=np.float32),
        100,
        200,
        200,
        100,
    )
    assert np.allclose(resized_boxes[0], np.array([20.0, 10.0, 80.0, 30.0], dtype=np.float32))

    letterboxed_boxes = rusty_cv.letterbox_boxes_numpy(
        np.array([[100.0, 50.0, 300.0, 150.0]], dtype=np.float32),
        400,
        200,
        640,
        640,
    )
    assert np.allclose(
        letterboxed_boxes[0],
        np.array([160.0, 240.0, 480.0, 400.0], dtype=np.float32),
    )
    assert np.allclose(
        rusty_cv.unletterbox_boxes_numpy(letterboxed_boxes, 400, 200, 640, 640),
        np.array([[100.0, 50.0, 300.0, 150.0]], dtype=np.float32),
    )

    scores = score_values
    class_ids = np.array([0, 0, 1], dtype=np.int64)
    class_scores = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.75],
            [0.1, 0.7],
        ],
        dtype=np.float32,
    )

    keep = rusty_cv.nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]
    filtered_keep = rusty_cv.nms(
        boxes,
        scores,
        iou_threshold=0.5,
        score_threshold=0.75,
        pre_nms_top_k=2,
        max_detections=1,
    )
    assert filtered_keep == [0]

    batched = rusty_cv.batched_nms(
        boxes,
        scores,
        class_ids,
        iou_threshold=0.5,
    )
    assert batched["indices"].tolist() == [0, 2]
    assert batched["class_ids"].tolist() == [0, 1]
    assert np.allclose(batched["scores"], np.array([0.9, 0.7], dtype=np.float32))

    multiclass = rusty_cv.multiclass_nms(
        boxes,
        class_scores,
        iou_threshold=0.5,
        score_threshold=0.7,
        max_detections=3,
    )
    assert multiclass["indices"].tolist() == [0, 1, 2]
    assert multiclass["class_ids"].tolist() == [0, 1, 1]
    assert np.allclose(multiclass["scores"], np.array([0.9, 0.75, 0.7], dtype=np.float32))

    soft_linear = rusty_cv.soft_nms(
        boxes,
        scores,
        method="linear",
        iou_threshold=0.5,
        score_threshold=0.2,
    )
    assert soft_linear["indices"].tolist() == [0, 2, 1]
    assert np.isclose(float(soft_linear["scores"][0]), 0.9)
    assert np.isclose(float(soft_linear["scores"][2]), 0.25546217, atol=1e-6)

    soft_gaussian = rusty_cv.soft_nms(
        boxes[:2],
        scores[:2],
        method="gaussian",
        iou_threshold=0.1,
        score_threshold=0.2,
        sigma=0.5,
    )
    assert soft_gaussian["indices"].tolist() == [0, 1]
    assert np.isclose(float(soft_gaussian["scores"][1]), 0.31670862, atol=1e-6)

    batched_soft = rusty_cv.batched_soft_nms(
        boxes,
        scores,
        class_ids,
        method="linear",
        iou_threshold=0.5,
        score_threshold=0.2,
    )
    assert batched_soft["indices"].tolist() == [0, 2, 1]
    assert batched_soft["class_ids"].tolist() == [0, 1, 0]
    assert np.isclose(float(batched_soft["scores"][2]), 0.25546217, atol=1e-6)

    multiclass_soft = rusty_cv.multiclass_soft_nms(
        boxes,
        class_scores,
        method="linear",
        iou_threshold=0.5,
        score_threshold=0.25,
        max_detections=4,
    )
    assert multiclass_soft["indices"].tolist() == [0, 1, 2, 1]
    assert multiclass_soft["class_ids"].tolist() == [0, 1, 1, 0]
    assert np.isclose(float(multiclass_soft["scores"][3]), 0.25546217, atol=1e-6)
    assert np.isclose(rusty_cv.iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)), 25.0 / 175.0)

    print("python smoke test: ok")


if __name__ == "__main__":
    main()
