use super::*;

#[test]
fn computes_iou() {
    let a = BBoxXYXY {
        x1: 0.0,
        y1: 0.0,
        x2: 10.0,
        y2: 10.0,
    };
    let b = BBoxXYXY {
        x1: 5.0,
        y1: 5.0,
        x2: 15.0,
        y2: 15.0,
    };

    assert!((iou(a, b) - (25.0 / 175.0)).abs() < 1e-6);
}

#[test]
fn converts_between_xyxy_and_xywh() {
    let boxes = vec![BBoxXYXY::from_xywh(2.0, 3.0, 4.0, 5.0)];
    let xywh = xyxy_to_xywh(&boxes).unwrap();
    assert_eq!(
        xywh,
        vec![BBoxXYWH {
            x: 2.0,
            y: 3.0,
            width: 4.0,
            height: 5.0,
        }]
    );
    assert_eq!(xywh_to_xyxy(&xywh).unwrap(), boxes);
}

#[test]
fn clips_boxes_to_image_bounds() {
    let boxes = vec![BBoxXYXY {
        x1: -5.0,
        y1: 3.0,
        x2: 12.0,
        y2: 30.0,
    }];

    let clipped = clip_boxes(&boxes, 10, 20).unwrap();

    assert_eq!(
        clipped,
        vec![BBoxXYXY {
            x1: 0.0,
            y1: 3.0,
            x2: 10.0,
            y2: 20.0,
        }]
    );
}

#[test]
fn rescales_boxes_for_direct_resize() {
    let boxes = vec![BBoxXYXY::from_xywh(10.0, 20.0, 30.0, 40.0)];

    let resized = resize_boxes(&boxes, 100, 200, 200, 100).unwrap();

    assert_eq!(
        resized,
        vec![BBoxXYXY {
            x1: 20.0,
            y1: 10.0,
            x2: 80.0,
            y2: 30.0,
        }]
    );
}

#[test]
fn remaps_boxes_through_letterbox_and_back() {
    let boxes = vec![BBoxXYXY::from_xywh(100.0, 50.0, 200.0, 100.0)];

    let letterboxed = letterbox_boxes(&boxes, 400, 200, 640, 640).unwrap();
    assert_eq!(
        letterboxed,
        vec![BBoxXYXY {
            x1: 160.0,
            y1: 240.0,
            x2: 480.0,
            y2: 400.0,
        }]
    );

    let restored = unletterbox_boxes(&letterboxed, 400, 200, 640, 640).unwrap();
    assert_eq!(restored, boxes);
}

#[test]
fn filters_boxes_by_score() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(10.0, 10.0, 4.0, 4.0),
        BBoxXYXY::from_xywh(20.0, 20.0, 2.0, 2.0),
    ];
    let scores = vec![0.9, 0.4, 0.7];

    let kept = filter_boxes_by_score(&boxes, &scores, 0.5).unwrap();

    assert_eq!(kept, vec![0, 2]);
}

#[test]
fn filters_boxes_by_area() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(10.0, 10.0, 4.0, 4.0),
        BBoxXYXY::from_xywh(20.0, 20.0, 2.0, 2.0),
    ];

    let kept = filter_boxes_by_area(&boxes, Some(10.0), Some(20.0)).unwrap();

    assert_eq!(kept, vec![1]);
}

#[test]
fn filters_boxes_by_min_size() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(10.0, 10.0, 4.0, 6.0),
        BBoxXYXY::from_xywh(20.0, 20.0, 2.0, 8.0),
    ];

    let kept = filter_boxes_by_min_size(&boxes, 4.0, 7.0).unwrap();

    assert_eq!(kept, vec![0]);
}

#[test]
fn clips_and_filters_boxes() {
    let boxes = vec![
        BBoxXYXY::from_xywh(-4.0, 1.0, 8.0, 8.0),
        BBoxXYXY::from_xywh(2.0, 2.0, 1.0, 1.0),
        BBoxXYXY::from_xywh(8.0, 8.0, 6.0, 6.0),
    ];

    let result = clip_and_filter_boxes(&boxes, 10, 10, 2.0, 2.0).unwrap();

    assert_eq!(result.indices, vec![0, 2]);
    assert_eq!(
        result.boxes,
        vec![
            BBoxXYXY {
                x1: 0.0,
                y1: 1.0,
                x2: 4.0,
                y2: 9.0,
            },
            BBoxXYXY {
                x1: 8.0,
                y1: 8.0,
                x2: 10.0,
                y2: 10.0,
            },
        ]
    );
}

#[test]
fn postprocesses_current_space_boxes() {
    let boxes = vec![
        BBoxXYXY::from_xywh(-5.0, 0.0, 8.0, 8.0),
        BBoxXYXY::from_xywh(0.0, 0.0, 1.0, 1.0),
        BBoxXYXY::from_xywh(2.0, 2.0, 4.0, 4.0),
    ];
    let scores = vec![0.9, 0.8, 0.7];
    let class_ids = vec![0usize, 0usize, 0usize];
    let options = PostprocessOptions {
        min_width: 2.0,
        min_height: 2.0,
        clip: true,
        ..PostprocessOptions::default()
    };

    let result = postprocess_detections(
        &boxes,
        &scores,
        &class_ids,
        BoxRemap::Current {
            width: 4,
            height: 4,
        },
        &options,
        None,
    )
    .unwrap();

    assert_eq!(result.detections.len(), 2);
    assert_eq!(result.detections[0].box_index, 0);
    assert_eq!(result.detections[1].box_index, 2);
    assert_eq!(
        result.boxes,
        vec![
            BBoxXYXY {
                x1: 0.0,
                y1: 0.0,
                x2: 3.0,
                y2: 4.0,
            },
            BBoxXYXY {
                x1: 2.0,
                y1: 2.0,
                x2: 4.0,
                y2: 4.0,
            },
        ]
    );
}

#[test]
fn postprocesses_letterboxed_boxes_back_to_original_space() {
    let boxes = vec![
        BBoxXYXY {
            x1: 160.0,
            y1: 240.0,
            x2: 480.0,
            y2: 400.0,
        },
        BBoxXYXY {
            x1: 170.0,
            y1: 250.0,
            x2: 490.0,
            y2: 410.0,
        },
        BBoxXYXY {
            x1: 100.0,
            y1: 240.0,
            x2: 140.0,
            y2: 280.0,
        },
    ];
    let scores = vec![0.95, 0.90, 0.70];
    let class_ids = vec![0usize, 0usize, 1usize];

    let result = postprocess_detections(
        &boxes,
        &scores,
        &class_ids,
        BoxRemap::Letterbox {
            processed_width: 640,
            processed_height: 640,
            original_width: 400,
            original_height: 200,
        },
        &PostprocessOptions {
            clip: true,
            ..PostprocessOptions::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(result.detections.len(), 2);
    assert_eq!(result.detections[0].box_index, 0);
    assert_eq!(result.detections[1].box_index, 2);
    assert_eq!(
        result.boxes,
        vec![
            BBoxXYXY {
                x1: 100.0,
                y1: 50.0,
                x2: 300.0,
                y2: 150.0,
            },
            BBoxXYXY {
                x1: 62.5,
                y1: 50.0,
                x2: 87.5,
                y2: 75.0,
            },
        ]
    );
}

#[test]
fn keeps_highest_scoring_boxes() {
    let boxes = vec![
        BBoxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        },
        BBoxXYXY {
            x1: 1.0,
            y1: 1.0,
            x2: 11.0,
            y2: 11.0,
        },
        BBoxXYXY {
            x1: 20.0,
            y1: 20.0,
            x2: 30.0,
            y2: 30.0,
        },
    ];
    let scores = vec![0.9, 0.8, 0.7];

    let keep = nms(&boxes, &scores, 0.5).unwrap();

    assert_eq!(keep, vec![0, 2]);
}

#[test]
fn applies_single_class_options() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(30.0, 30.0, 8.0, 8.0),
    ];
    let scores = vec![0.95, 0.90, 0.40];
    let options = NmsOptions {
        iou_threshold: 0.5,
        score_threshold: 0.5,
        pre_nms_top_k: Some(2),
        max_detections: Some(1),
    };

    let keep = nms_with_options(&boxes, &scores, &options).unwrap();

    assert_eq!(keep, vec![0]);
}

#[test]
fn batched_nms_keeps_overlapping_boxes_from_different_classes() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(0.5, 0.5, 10.0, 10.0),
        BBoxXYXY::from_xywh(25.0, 25.0, 5.0, 5.0),
    ];
    let scores = vec![0.95, 0.90, 0.92, 0.80];
    let class_ids = vec![0usize, 0usize, 1usize, 1usize];

    let detections = batched_nms(&boxes, &scores, &class_ids, &NmsOptions::default()).unwrap();

    assert_eq!(
        detections,
        vec![
            Detection {
                box_index: 0,
                class_id: 0,
                score: 0.95,
            },
            Detection {
                box_index: 2,
                class_id: 1,
                score: 0.92,
            },
            Detection {
                box_index: 3,
                class_id: 1,
                score: 0.80,
            },
        ]
    );
}

#[test]
fn multiclass_nms_expands_scores_per_class() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(20.0, 20.0, 6.0, 6.0),
    ];
    let class_scores = vec![0.95, 0.10, 0.90, 0.85, 0.40, 0.80];
    let options = NmsOptions {
        iou_threshold: 0.5,
        score_threshold: 0.5,
        pre_nms_top_k: None,
        max_detections: Some(3),
    };

    let detections = multiclass_nms(&boxes, &class_scores, 2, &options).unwrap();

    assert_eq!(
        detections,
        vec![
            Detection {
                box_index: 0,
                class_id: 0,
                score: 0.95,
            },
            Detection {
                box_index: 1,
                class_id: 1,
                score: 0.85,
            },
            Detection {
                box_index: 2,
                class_id: 1,
                score: 0.80,
            },
        ]
    );
}

#[test]
fn soft_nms_linear_decays_scores() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
    ];
    let scores = vec![0.9, 0.8];
    let options = SoftNmsOptions {
        method: SoftNmsMethod::Linear,
        iou_threshold: 0.5,
        score_threshold: 0.2,
        sigma: 0.5,
        pre_nms_top_k: None,
        max_detections: None,
    };

    let detections = soft_nms(&boxes, &scores, &options).unwrap();

    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0].box_index, 0);
    assert!((detections[0].score - 0.9).abs() < 1e-6);
    assert_eq!(detections[1].box_index, 1);
    assert!((detections[1].score - 0.25546217).abs() < 1e-6);
}

#[test]
fn soft_nms_gaussian_uses_sigma_decay() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
    ];
    let scores = vec![0.9, 0.8];
    let options = SoftNmsOptions {
        method: SoftNmsMethod::Gaussian,
        iou_threshold: 0.1,
        score_threshold: 0.2,
        sigma: 0.5,
        pre_nms_top_k: None,
        max_detections: None,
    };

    let detections = soft_nms(&boxes, &scores, &options).unwrap();

    assert_eq!(detections.len(), 2);
    assert_eq!(detections[1].box_index, 1);
    assert!((detections[1].score - 0.31670862).abs() < 1e-6);
}

#[test]
fn soft_nms_respects_threshold_and_top_k() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(30.0, 30.0, 6.0, 6.0),
    ];
    let scores = vec![0.9, 0.8, 0.7];
    let options = SoftNmsOptions {
        method: SoftNmsMethod::Linear,
        iou_threshold: 0.5,
        score_threshold: 0.3,
        sigma: 0.5,
        pre_nms_top_k: Some(2),
        max_detections: None,
    };

    let detections = soft_nms(&boxes, &scores, &options).unwrap();

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].box_index, 0);
}

#[test]
fn batched_soft_nms_keeps_classes_separate() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(0.5, 0.5, 10.0, 10.0),
    ];
    let scores = vec![0.9, 0.8, 0.85];
    let class_ids = vec![0usize, 0usize, 1usize];
    let options = SoftNmsOptions {
        method: SoftNmsMethod::Linear,
        iou_threshold: 0.5,
        score_threshold: 0.2,
        sigma: 0.5,
        pre_nms_top_k: None,
        max_detections: None,
    };

    let detections = batched_soft_nms(&boxes, &scores, &class_ids, &options).unwrap();

    assert_eq!(
        detections,
        vec![
            Detection {
                box_index: 0,
                class_id: 0,
                score: 0.9,
            },
            Detection {
                box_index: 2,
                class_id: 1,
                score: 0.85,
            },
            Detection {
                box_index: 1,
                class_id: 0,
                score: 0.25546217,
            },
        ]
    );
}

#[test]
fn multiclass_soft_nms_expands_scores_per_class() {
    let boxes = vec![
        BBoxXYXY::from_xywh(0.0, 0.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(1.0, 1.0, 10.0, 10.0),
        BBoxXYXY::from_xywh(20.0, 20.0, 6.0, 6.0),
    ];
    let class_scores = vec![0.95, 0.10, 0.90, 0.85, 0.40, 0.80];
    let options = SoftNmsOptions {
        method: SoftNmsMethod::Linear,
        iou_threshold: 0.5,
        score_threshold: 0.25,
        sigma: 0.5,
        pre_nms_top_k: None,
        max_detections: Some(4),
    };

    let detections = multiclass_soft_nms(&boxes, &class_scores, 2, &options).unwrap();

    assert_eq!(detections.len(), 4);
    assert_eq!(detections[0].box_index, 0);
    assert_eq!(detections[0].class_id, 0);
    assert!((detections[0].score - 0.95).abs() < 1e-6);
    assert_eq!(detections[1].box_index, 1);
    assert_eq!(detections[1].class_id, 1);
    assert!((detections[1].score - 0.85).abs() < 1e-6);
    assert_eq!(detections[2].box_index, 2);
    assert_eq!(detections[2].class_id, 1);
    assert!((detections[2].score - 0.80).abs() < 1e-6);
    assert_eq!(detections[3].box_index, 2);
    assert_eq!(detections[3].class_id, 0);
    assert!((detections[3].score - 0.40).abs() < 1e-6);
}

#[test]
fn allows_same_boxes_when_threshold_is_one() {
    let boxes = vec![
        BBoxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        },
        BBoxXYXY {
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        },
    ];
    let scores = vec![0.8, 0.7];

    let keep = nms(&boxes, &scores, 1.0).unwrap();

    assert_eq!(keep, vec![0, 1]);
}

#[test]
fn rejects_invalid_inputs() {
    let boxes = vec![BBoxXYXY {
        x1: 0.0,
        y1: 0.0,
        x2: 1.0,
        y2: 1.0,
    }];

    assert_eq!(
        nms(&boxes, &[], 0.5).unwrap_err(),
        BBoxError::LengthMismatch {
            boxes: 1,
            scores: 0
        }
    );
    assert_eq!(
        nms(&boxes, &[0.5], 1.5).unwrap_err(),
        BBoxError::InvalidIouThreshold(1.5)
    );
    assert_eq!(
        batched_nms(&boxes, &[0.5], &[], &NmsOptions::default()).unwrap_err(),
        BBoxError::ClassLengthMismatch {
            boxes: 1,
            class_ids: 0,
        }
    );
    assert_eq!(
        multiclass_nms(&boxes, &[0.5, 0.4], 0, &NmsOptions::default()).unwrap_err(),
        BBoxError::InvalidNumClasses(0)
    );
    assert_eq!(
        multiclass_nms(&boxes, &[0.5], 2, &NmsOptions::default()).unwrap_err(),
        BBoxError::ClassScoreShapeMismatch {
            boxes: 1,
            class_scores: 1,
            num_classes: 2,
        }
    );
    assert_eq!(
        clip_boxes(&boxes, 0, 10).unwrap_err(),
        BBoxError::InvalidImageSize {
            width: 0,
            height: 10,
        }
    );
    assert!(matches!(
        filter_boxes_by_score(&boxes, &[0.5], f32::NAN).unwrap_err(),
        BBoxError::InvalidScoreThreshold(value) if value.is_nan()
    ));
    assert_eq!(
        filter_boxes_by_area(&boxes, Some(-1.0), None).unwrap_err(),
        BBoxError::InvalidMinArea(-1.0)
    );
    assert_eq!(
        filter_boxes_by_area(&boxes, Some(5.0), Some(4.0)).unwrap_err(),
        BBoxError::InvalidAreaRange {
            min_area: 5.0,
            max_area: 4.0,
        }
    );
    assert_eq!(
        filter_boxes_by_min_size(&boxes, -1.0, 0.0).unwrap_err(),
        BBoxError::InvalidMinSize {
            min_width: -1.0,
            min_height: 0.0,
        }
    );
    assert_eq!(
        soft_nms(
            &boxes,
            &[0.5],
            &SoftNmsOptions {
                sigma: 0.0,
                ..SoftNmsOptions::default()
            },
        )
        .unwrap_err(),
        BBoxError::InvalidSigma(0.0)
    );
    assert_eq!(
        batched_soft_nms(&boxes, &[0.5], &[], &SoftNmsOptions::default()).unwrap_err(),
        BBoxError::ClassLengthMismatch {
            boxes: 1,
            class_ids: 0,
        }
    );
    assert_eq!(
        multiclass_soft_nms(&boxes, &[0.5], 2, &SoftNmsOptions::default()).unwrap_err(),
        BBoxError::ClassScoreShapeMismatch {
            boxes: 1,
            class_scores: 1,
            num_classes: 2,
        }
    );
}
