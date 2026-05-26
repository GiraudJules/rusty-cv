use super::{BBoxError, BBoxXYXY, BoxFilterResult};

/// Clamp boxes to image bounds.
pub fn clip_boxes(boxes: &[BBoxXYXY], width: u32, height: u32) -> Result<Vec<BBoxXYXY>, BBoxError> {
    super::validate_boxes(boxes)?;
    super::validate_image_size(width, height)?;
    let width = width as f32;
    let height = height as f32;
    Ok(boxes.iter().map(|bbox| bbox.clip(width, height)).collect())
}

/// Keep the indices of boxes whose scores are above `threshold`.
pub fn filter_boxes_by_score(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    threshold: f32,
) -> Result<Vec<usize>, BBoxError> {
    if boxes.len() != scores.len() {
        return Err(BBoxError::LengthMismatch {
            boxes: boxes.len(),
            scores: scores.len(),
        });
    }

    if threshold.is_nan() {
        return Err(BBoxError::InvalidScoreThreshold(threshold));
    }

    super::validate_boxes(boxes)?;
    super::validate_scores(scores)?;
    Ok(scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| (*score >= threshold).then_some(index))
        .collect())
}

/// Keep the indices of boxes whose area falls within the requested range.
pub fn filter_boxes_by_area(
    boxes: &[BBoxXYXY],
    min_area: Option<f32>,
    max_area: Option<f32>,
) -> Result<Vec<usize>, BBoxError> {
    super::validate_boxes(boxes)?;
    super::validate_area_bounds(min_area, max_area)?;

    Ok(boxes
        .iter()
        .enumerate()
        .filter_map(|(index, bbox)| {
            let area = bbox.area();
            let keep_min = min_area.is_none_or(|value| area >= value);
            let keep_max = max_area.is_none_or(|value| area <= value);
            (keep_min && keep_max).then_some(index)
        })
        .collect())
}

/// Keep the indices of boxes whose width and height meet the requested minimums.
pub fn filter_boxes_by_min_size(
    boxes: &[BBoxXYXY],
    min_width: f32,
    min_height: f32,
) -> Result<Vec<usize>, BBoxError> {
    super::validate_boxes(boxes)?;
    super::validate_min_size(min_width, min_height)?;

    Ok(boxes
        .iter()
        .enumerate()
        .filter_map(|(index, bbox)| {
            (bbox.width() >= min_width && bbox.height() >= min_height).then_some(index)
        })
        .collect())
}

/// Clip boxes to image bounds, then keep only boxes whose clipped width and height are large enough.
pub fn clip_and_filter_boxes(
    boxes: &[BBoxXYXY],
    width: u32,
    height: u32,
    min_width: f32,
    min_height: f32,
) -> Result<BoxFilterResult, BBoxError> {
    super::validate_boxes(boxes)?;
    super::validate_image_size(width, height)?;
    super::validate_min_size(min_width, min_height)?;

    let clipped = clip_boxes(boxes, width, height)?;
    let mut filtered_boxes = Vec::new();
    let mut kept_indices = Vec::new();

    for (index, bbox) in clipped.into_iter().enumerate() {
        if bbox.width() >= min_width && bbox.height() >= min_height {
            filtered_boxes.push(bbox);
            kept_indices.push(index);
        }
    }

    Ok(BoxFilterResult {
        boxes: filtered_boxes,
        indices: kept_indices,
    })
}
