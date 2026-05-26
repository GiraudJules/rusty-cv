use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::{BBoxError, BBoxXYXY, Detection, NmsOptions, SoftNmsMethod, SoftNmsOptions};

/// Run single-class non-maximum suppression with custom options.
///
/// Returns the kept indices in descending score order.
pub fn nms_with_options(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    options: &NmsOptions,
) -> Result<Vec<usize>, BBoxError> {
    let class_ids = vec![0usize; boxes.len()];
    let detections = batched_nms(boxes, scores, &class_ids, options)?;
    Ok(detections
        .into_iter()
        .map(|detection| detection.box_index)
        .collect())
}

/// Run single-class non-maximum suppression.
///
/// Returns the kept indices in descending score order.
pub fn nms(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    iou_threshold: f32,
) -> Result<Vec<usize>, BBoxError> {
    let options = NmsOptions {
        iou_threshold,
        ..NmsOptions::default()
    };
    nms_with_options(boxes, scores, &options)
}

/// Run class-aware non-maximum suppression for one score and one class per box.
pub fn batched_nms(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    class_ids: &[usize],
    options: &NmsOptions,
) -> Result<Vec<Detection>, BBoxError> {
    super::validate_nms_inputs(boxes, scores, options)?;

    if boxes.len() != class_ids.len() {
        return Err(BBoxError::ClassLengthMismatch {
            boxes: boxes.len(),
            class_ids: class_ids.len(),
        });
    }

    let mut candidates_by_class: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..boxes.len() {
        if scores[index] >= options.score_threshold {
            candidates_by_class
                .entry(class_ids[index])
                .or_default()
                .push(index);
        }
    }

    let mut detections = Vec::new();
    for (class_id, candidate_indices) in &mut candidates_by_class {
        sort_indices_by_score_desc(candidate_indices, |index| scores[index]);

        if let Some(limit) = options.pre_nms_top_k {
            candidate_indices.truncate(limit);
        }

        let kept_indices = nms_from_sorted_indices(boxes, candidate_indices, options.iou_threshold);
        for box_index in kept_indices {
            detections.push(Detection {
                box_index,
                class_id: *class_id,
                score: scores[box_index],
            });
        }
    }

    sort_detections_desc(&mut detections);
    if let Some(limit) = options.max_detections {
        detections.truncate(limit);
    }

    Ok(detections)
}

/// Run multiclass non-maximum suppression over `num_classes` scores for each box.
pub fn multiclass_nms(
    boxes: &[BBoxXYXY],
    class_scores: &[f32],
    num_classes: usize,
    options: &NmsOptions,
) -> Result<Vec<Detection>, BBoxError> {
    super::validate_thresholds(options)?;
    super::validate_boxes(boxes)?;

    if num_classes == 0 {
        return Err(BBoxError::InvalidNumClasses(num_classes));
    }

    let expected_scores = boxes.len().saturating_mul(num_classes);
    if class_scores.len() != expected_scores {
        return Err(BBoxError::ClassScoreShapeMismatch {
            boxes: boxes.len(),
            class_scores: class_scores.len(),
            num_classes,
        });
    }

    super::validate_scores(class_scores)?;

    let mut detections = Vec::new();
    for class_id in 0..num_classes {
        let mut candidate_indices = Vec::new();
        for box_index in 0..boxes.len() {
            let score = class_scores[box_index * num_classes + class_id];
            if score >= options.score_threshold {
                candidate_indices.push(box_index);
            }
        }

        sort_indices_by_score_desc(&mut candidate_indices, |index| {
            class_scores[index * num_classes + class_id]
        });

        if let Some(limit) = options.pre_nms_top_k {
            candidate_indices.truncate(limit);
        }

        let kept_indices =
            nms_from_sorted_indices(boxes, &candidate_indices, options.iou_threshold);
        for box_index in kept_indices {
            detections.push(Detection {
                box_index,
                class_id,
                score: class_scores[box_index * num_classes + class_id],
            });
        }
    }

    sort_detections_desc(&mut detections);
    if let Some(limit) = options.max_detections {
        detections.truncate(limit);
    }

    Ok(detections)
}

/// Run single-class soft NMS and return detections with decayed scores.
pub fn soft_nms(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    options: &SoftNmsOptions,
) -> Result<Vec<Detection>, BBoxError> {
    if boxes.len() != scores.len() {
        return Err(BBoxError::LengthMismatch {
            boxes: boxes.len(),
            scores: scores.len(),
        });
    }

    super::validate_soft_nms_options(options)?;
    super::validate_boxes(boxes)?;
    super::validate_scores(scores)?;

    let mut candidate_indices: Vec<usize> = (0..boxes.len()).collect();
    sort_indices_by_score_desc(&mut candidate_indices, |index| scores[index]);

    if let Some(limit) = options.pre_nms_top_k {
        candidate_indices.truncate(limit);
    }

    let candidates = candidate_indices
        .into_iter()
        .map(|box_index| Detection {
            box_index,
            class_id: 0,
            score: scores[box_index],
        })
        .collect::<Vec<_>>();
    let mut detections = soft_nms_from_candidates(boxes, candidates, options);
    if let Some(limit) = options.max_detections {
        detections.truncate(limit);
    }

    Ok(detections)
}

/// Run class-aware soft NMS for one score and one class per box.
pub fn batched_soft_nms(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    class_ids: &[usize],
    options: &SoftNmsOptions,
) -> Result<Vec<Detection>, BBoxError> {
    if boxes.len() != scores.len() {
        return Err(BBoxError::LengthMismatch {
            boxes: boxes.len(),
            scores: scores.len(),
        });
    }

    if boxes.len() != class_ids.len() {
        return Err(BBoxError::ClassLengthMismatch {
            boxes: boxes.len(),
            class_ids: class_ids.len(),
        });
    }

    super::validate_soft_nms_options(options)?;
    super::validate_boxes(boxes)?;
    super::validate_scores(scores)?;

    let mut candidates_by_class: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..boxes.len() {
        if scores[index] >= options.score_threshold {
            candidates_by_class
                .entry(class_ids[index])
                .or_default()
                .push(index);
        }
    }

    let mut detections = Vec::new();
    for (class_id, candidate_indices) in &mut candidates_by_class {
        sort_indices_by_score_desc(candidate_indices, |index| scores[index]);
        if let Some(limit) = options.pre_nms_top_k {
            candidate_indices.truncate(limit);
        }

        let candidates = candidate_indices
            .iter()
            .copied()
            .map(|box_index| Detection {
                box_index,
                class_id: *class_id,
                score: scores[box_index],
            })
            .collect::<Vec<_>>();
        detections.extend(soft_nms_from_candidates(boxes, candidates, options));
    }

    sort_detections_desc(&mut detections);
    if let Some(limit) = options.max_detections {
        detections.truncate(limit);
    }

    Ok(detections)
}

/// Run multiclass soft NMS over `num_classes` scores for each box.
pub fn multiclass_soft_nms(
    boxes: &[BBoxXYXY],
    class_scores: &[f32],
    num_classes: usize,
    options: &SoftNmsOptions,
) -> Result<Vec<Detection>, BBoxError> {
    super::validate_soft_nms_options(options)?;
    super::validate_boxes(boxes)?;

    if num_classes == 0 {
        return Err(BBoxError::InvalidNumClasses(num_classes));
    }

    let expected_scores = boxes.len().saturating_mul(num_classes);
    if class_scores.len() != expected_scores {
        return Err(BBoxError::ClassScoreShapeMismatch {
            boxes: boxes.len(),
            class_scores: class_scores.len(),
            num_classes,
        });
    }

    super::validate_scores(class_scores)?;

    let mut detections = Vec::new();
    for class_id in 0..num_classes {
        let mut candidate_indices = Vec::new();
        for box_index in 0..boxes.len() {
            let score = class_scores[box_index * num_classes + class_id];
            if score >= options.score_threshold {
                candidate_indices.push(box_index);
            }
        }

        sort_indices_by_score_desc(&mut candidate_indices, |index| {
            class_scores[index * num_classes + class_id]
        });
        if let Some(limit) = options.pre_nms_top_k {
            candidate_indices.truncate(limit);
        }

        let candidates = candidate_indices
            .into_iter()
            .map(|box_index| Detection {
                box_index,
                class_id,
                score: class_scores[box_index * num_classes + class_id],
            })
            .collect::<Vec<_>>();
        detections.extend(soft_nms_from_candidates(boxes, candidates, options));
    }

    sort_detections_desc(&mut detections);
    if let Some(limit) = options.max_detections {
        detections.truncate(limit);
    }

    Ok(detections)
}

fn sort_indices_by_score_desc<F>(indices: &mut [usize], score_for: F)
where
    F: Copy + Fn(usize) -> f32,
{
    indices.sort_by(|&left, &right| {
        score_for(right)
            .partial_cmp(&score_for(left))
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
}

fn sort_detections_desc(detections: &mut [Detection]) {
    detections.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.box_index.cmp(&right.box_index))
            .then_with(|| left.class_id.cmp(&right.class_id))
    });
}

fn soft_nms_from_candidates(
    boxes: &[BBoxXYXY],
    mut candidates: Vec<Detection>,
    options: &SoftNmsOptions,
) -> Vec<Detection> {
    let mut detections = Vec::with_capacity(candidates.len());

    while !candidates.is_empty() {
        let best_position = best_detection_position(&candidates);
        let best_detection = candidates.swap_remove(best_position);

        if best_detection.score < options.score_threshold {
            break;
        }

        detections.push(best_detection);

        let reference_box = boxes[best_detection.box_index];
        for detection in &mut candidates {
            let candidate_box = boxes[detection.box_index];
            let overlap = reference_box.iou(&candidate_box);
            let weight = soft_nms_weight(overlap, options);
            detection.score *= weight;
        }

        candidates.retain(|detection| detection.score >= options.score_threshold);
    }

    detections
}

fn best_detection_position(detections: &[Detection]) -> usize {
    let mut best_position = 0usize;
    for position in 1..detections.len() {
        let current = detections[position];
        let best = detections[best_position];
        let is_better = current
            .score
            .partial_cmp(&best.score)
            .unwrap_or(Ordering::Equal)
            == Ordering::Greater
            || (current.score == best.score && current.box_index < best.box_index);

        if is_better {
            best_position = position;
        }
    }
    best_position
}

fn soft_nms_weight(overlap: f32, options: &SoftNmsOptions) -> f32 {
    match options.method {
        SoftNmsMethod::Linear => {
            if overlap > options.iou_threshold {
                1.0 - overlap
            } else {
                1.0
            }
        }
        SoftNmsMethod::Gaussian => (-(overlap * overlap) / options.sigma).exp(),
    }
}

fn nms_from_sorted_indices(
    boxes: &[BBoxXYXY],
    sorted_indices: &[usize],
    iou_threshold: f32,
) -> Vec<usize> {
    let mut keep = Vec::with_capacity(sorted_indices.len());

    'candidate: for &index in sorted_indices {
        for &kept_index in &keep {
            if boxes[index].iou(&boxes[kept_index]) > iou_threshold {
                continue 'candidate;
            }
        }
        keep.push(index);
    }

    keep
}
