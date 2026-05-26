use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::letterbox::compute_letterbox;

/// Axis-aligned bounding box in `xyxy` format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBoxXYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Axis-aligned bounding box in `xywh` format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBoxXYWH {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BBoxXYXY {
    /// Create a box from `xywh` format.
    pub fn from_xywh(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x1: x,
            y1: y,
            x2: x + width,
            y2: y + height,
        }
    }

    /// Convert an `xyxy` box into `xywh` format.
    pub fn to_xywh(&self) -> BBoxXYWH {
        BBoxXYWH {
            x: self.x1,
            y: self.y1,
            width: self.x2 - self.x1,
            height: self.y2 - self.y1,
        }
    }

    /// Return the width of the box, clamped at zero for invalid geometry.
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Return the height of the box, clamped at zero for invalid geometry.
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Return the area of the box.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Return the intersection area between two boxes.
    pub fn intersection_area(&self, other: &Self) -> f32 {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);

        let width = (x2 - x1).max(0.0);
        let height = (y2 - y1).max(0.0);
        width * height
    }

    /// Return the union area between two boxes.
    pub fn union_area(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Return the intersection-over-union between two boxes.
    pub fn iou(&self, other: &Self) -> f32 {
        let union = self.union_area(other);
        if union <= 0.0 {
            0.0
        } else {
            self.intersection_area(other) / union
        }
    }

    fn is_finite(&self) -> bool {
        self.x1.is_finite() && self.y1.is_finite() && self.x2.is_finite() && self.y2.is_finite()
    }

    /// Clamp a box to image bounds in-place.
    pub fn clip(&self, width: f32, height: f32) -> Self {
        Self {
            x1: self.x1.clamp(0.0, width),
            y1: self.y1.clamp(0.0, height),
            x2: self.x2.clamp(0.0, width),
            y2: self.y2.clamp(0.0, height),
        }
    }
}

impl BBoxXYWH {
    /// Convert an `xywh` box into `xyxy` format.
    pub fn to_xyxy(&self) -> BBoxXYXY {
        BBoxXYXY::from_xywh(self.x, self.y, self.width, self.height)
    }
}

/// Options for NMS-style postprocessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NmsOptions {
    pub iou_threshold: f32,
    pub score_threshold: f32,
    pub pre_nms_top_k: Option<usize>,
    pub max_detections: Option<usize>,
}

impl Default for NmsOptions {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            score_threshold: f32::NEG_INFINITY,
            pre_nms_top_k: None,
            max_detections: None,
        }
    }
}

/// Scoring strategy for soft NMS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftNmsMethod {
    Linear,
    Gaussian,
}

/// Options for soft NMS postprocessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftNmsOptions {
    pub method: SoftNmsMethod,
    pub iou_threshold: f32,
    pub score_threshold: f32,
    pub sigma: f32,
    pub pre_nms_top_k: Option<usize>,
    pub max_detections: Option<usize>,
}

impl Default for SoftNmsOptions {
    fn default() -> Self {
        Self {
            method: SoftNmsMethod::Linear,
            iou_threshold: 0.5,
            score_threshold: f32::NEG_INFINITY,
            sigma: 0.5,
            pre_nms_top_k: None,
            max_detections: None,
        }
    }
}

/// Result item returned by batched and multiclass NMS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub box_index: usize,
    pub class_id: usize,
    pub score: f32,
}

/// Result returned by box filtering helpers that transform box geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxFilterResult {
    pub boxes: Vec<BBoxXYXY>,
    pub indices: Vec<usize>,
}

/// Geometry remapping mode for fused detection postprocessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoxRemap {
    None,
    Current {
        width: u32,
        height: u32,
    },
    Resize {
        processed_width: u32,
        processed_height: u32,
        original_width: u32,
        original_height: u32,
    },
    Letterbox {
        processed_width: u32,
        processed_height: u32,
        original_width: u32,
        original_height: u32,
    },
}

/// Options for fused detection postprocessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostprocessOptions {
    pub iou_threshold: f32,
    pub score_threshold: f32,
    pub pre_nms_top_k: Option<usize>,
    pub max_detections: Option<usize>,
    pub min_width: f32,
    pub min_height: f32,
    pub clip: bool,
}

impl Default for PostprocessOptions {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            score_threshold: f32::NEG_INFINITY,
            pre_nms_top_k: None,
            max_detections: None,
            min_width: 0.0,
            min_height: 0.0,
            clip: false,
        }
    }
}

/// Result returned by fused detection postprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct PostprocessResult {
    pub boxes: Vec<BBoxXYXY>,
    pub detections: Vec<Detection>,
}
/// Errors for box postprocessing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BBoxError {
    LengthMismatch {
        boxes: usize,
        scores: usize,
    },
    ClassLengthMismatch {
        boxes: usize,
        class_ids: usize,
    },
    ClassScoreShapeMismatch {
        boxes: usize,
        class_scores: usize,
        num_classes: usize,
    },
    InvalidIouThreshold(f32),
    InvalidScoreThreshold(f32),
    InvalidSigma(f32),
    InvalidNumClasses(usize),
    InvalidImageSize {
        width: u32,
        height: u32,
    },
    InvalidMinArea(f32),
    InvalidMaxArea(f32),
    InvalidAreaRange {
        min_area: f32,
        max_area: f32,
    },
    InvalidMinSize {
        min_width: f32,
        min_height: f32,
    },
    NonFiniteBox {
        index: usize,
    },
    NonFiniteScore {
        index: usize,
        value: f32,
    },
}

impl std::fmt::Display for BBoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { boxes, scores } => write!(
                f,
                "boxes and scores must have the same length, got {} boxes and {} scores",
                boxes, scores
            ),
            Self::ClassLengthMismatch { boxes, class_ids } => write!(
                f,
                "boxes and class_ids must have the same length, got {} boxes and {} class_ids",
                boxes, class_ids
            ),
            Self::ClassScoreShapeMismatch {
                boxes,
                class_scores,
                num_classes,
            } => write!(
                f,
                "class_scores must contain boxes * num_classes values, got {} boxes, {} class_scores values, and {} classes",
                boxes, class_scores, num_classes
            ),
            Self::InvalidIouThreshold(value) => write!(
                f,
                "iou_threshold must be in the inclusive range [0.0, 1.0], got {}",
                value
            ),
            Self::InvalidScoreThreshold(value) => {
                write!(f, "score_threshold must not be NaN, got {}", value)
            }
            Self::InvalidSigma(value) => {
                write!(f, "sigma must be finite and greater than zero, got {}", value)
            }
            Self::InvalidNumClasses(value) => {
                write!(f, "num_classes must be greater than zero, got {}", value)
            }
            Self::InvalidImageSize { width, height } => write!(
                f,
                "image width and height must be greater than zero, got {}x{}",
                width, height
            ),
            Self::InvalidMinArea(value) => {
                write!(f, "min_area must be finite and non-negative, got {}", value)
            }
            Self::InvalidMaxArea(value) => {
                write!(f, "max_area must be finite and non-negative, got {}", value)
            }
            Self::InvalidAreaRange { min_area, max_area } => write!(
                f,
                "max_area must be greater than or equal to min_area, got min_area={} and max_area={}",
                min_area, max_area
            ),
            Self::InvalidMinSize {
                min_width,
                min_height,
            } => write!(
                f,
                "min_width and min_height must be finite and non-negative, got {} and {}",
                min_width, min_height
            ),
            Self::NonFiniteBox { index } => {
                write!(f, "box at index {} contains a non-finite coordinate", index)
            }
            Self::NonFiniteScore { index, value } => write!(
                f,
                "score at index {} must be finite, got {}",
                index, value
            ),
        }
    }
}

impl std::error::Error for BBoxError {}

/// Compute IoU between two `xyxy` boxes.
pub fn iou(a: BBoxXYXY, b: BBoxXYXY) -> f32 {
    a.iou(&b)
}

/// Convert boxes from `xyxy` into `xywh` format.
pub fn xyxy_to_xywh(boxes: &[BBoxXYXY]) -> Result<Vec<BBoxXYWH>, BBoxError> {
    validate_boxes(boxes)?;
    Ok(boxes.iter().map(BBoxXYXY::to_xywh).collect())
}

/// Convert boxes from `xywh` into `xyxy` format.
pub fn xywh_to_xyxy(boxes: &[BBoxXYWH]) -> Result<Vec<BBoxXYXY>, BBoxError> {
    for (index, bbox) in boxes.iter().enumerate() {
        let is_finite = bbox.x.is_finite()
            && bbox.y.is_finite()
            && bbox.width.is_finite()
            && bbox.height.is_finite();
        if !is_finite {
            return Err(BBoxError::NonFiniteBox { index });
        }
    }

    Ok(boxes.iter().map(BBoxXYWH::to_xyxy).collect())
}

/// Clamp boxes to image bounds.
pub fn clip_boxes(boxes: &[BBoxXYXY], width: u32, height: u32) -> Result<Vec<BBoxXYXY>, BBoxError> {
    validate_boxes(boxes)?;
    validate_image_size(width, height)?;
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

    validate_boxes(boxes)?;
    validate_scores(scores)?;
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
    validate_boxes(boxes)?;
    validate_area_bounds(min_area, max_area)?;

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
    validate_boxes(boxes)?;
    validate_min_size(min_width, min_height)?;

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
    validate_boxes(boxes)?;
    validate_image_size(width, height)?;
    validate_min_size(min_width, min_height)?;

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

/// Map boxes from one image size to another with direct resize scaling.
pub fn resize_boxes(
    boxes: &[BBoxXYXY],
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<BBoxXYXY>, BBoxError> {
    validate_boxes(boxes)?;
    validate_image_size(original_width, original_height)?;
    validate_image_size(target_width, target_height)?;

    let scale_x = target_width as f32 / original_width as f32;
    let scale_y = target_height as f32 / original_height as f32;

    Ok(boxes
        .iter()
        .map(|bbox| BBoxXYXY {
            x1: bbox.x1 * scale_x,
            y1: bbox.y1 * scale_y,
            x2: bbox.x2 * scale_x,
            y2: bbox.y2 * scale_y,
        })
        .collect())
}

/// Map boxes from original image space into letterboxed image space.
pub fn letterbox_boxes(
    boxes: &[BBoxXYXY],
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<BBoxXYXY>, BBoxError> {
    validate_boxes(boxes)?;
    validate_image_size(original_width, original_height)?;
    validate_image_size(target_width, target_height)?;
    let info = compute_letterbox(original_width, original_height, target_width, target_height)
        .expect("validated image sizes should satisfy compute_letterbox");

    let scale = info.scale;
    let pad_x = info.padding.left as f32;
    let pad_y = info.padding.top as f32;

    Ok(boxes
        .iter()
        .map(|bbox| BBoxXYXY {
            x1: bbox.x1 * scale + pad_x,
            y1: bbox.y1 * scale + pad_y,
            x2: bbox.x2 * scale + pad_x,
            y2: bbox.y2 * scale + pad_y,
        })
        .collect())
}

/// Map boxes from letterboxed image space back to the original image space.
pub fn unletterbox_boxes(
    boxes: &[BBoxXYXY],
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<BBoxXYXY>, BBoxError> {
    validate_boxes(boxes)?;
    validate_image_size(original_width, original_height)?;
    validate_image_size(target_width, target_height)?;
    let info = compute_letterbox(original_width, original_height, target_width, target_height)
        .expect("validated image sizes should satisfy compute_letterbox");

    let scale = info.scale;
    let pad_x = info.padding.left as f32;
    let pad_y = info.padding.top as f32;

    Ok(boxes
        .iter()
        .map(|bbox| BBoxXYXY {
            x1: (bbox.x1 - pad_x) / scale,
            y1: (bbox.y1 - pad_y) / scale,
            x2: (bbox.x2 - pad_x) / scale,
            y2: (bbox.y2 - pad_y) / scale,
        })
        .collect())
}

/// Run fused detection postprocessing over remapped boxes and class-aware NMS.
pub fn postprocess_detections(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    class_ids: &[usize],
    remap: BoxRemap,
    options: &PostprocessOptions,
    soft_nms_options: Option<&SoftNmsOptions>,
) -> Result<PostprocessResult, BBoxError> {
    validate_postprocess_inputs(boxes, scores, class_ids, options)?;

    let (mut remapped_boxes, clip_bounds) = remap_boxes_for_postprocess(boxes, remap)?;
    if options.clip {
        if let Some((width, height)) = clip_bounds {
            remapped_boxes = clip_boxes(&remapped_boxes, width, height)?;
        }
    }

    let candidate_indices =
        filter_boxes_by_min_size(&remapped_boxes, options.min_width, options.min_height)?;
    let candidate_boxes = candidate_indices
        .iter()
        .map(|&index| remapped_boxes[index])
        .collect::<Vec<_>>();
    let candidate_scores = candidate_indices
        .iter()
        .map(|&index| scores[index])
        .collect::<Vec<_>>();
    let candidate_class_ids = candidate_indices
        .iter()
        .map(|&index| class_ids[index])
        .collect::<Vec<_>>();

    let detections = if let Some(soft_options) = soft_nms_options {
        batched_soft_nms(
            &candidate_boxes,
            &candidate_scores,
            &candidate_class_ids,
            soft_options,
        )?
    } else {
        let nms_options = NmsOptions {
            iou_threshold: options.iou_threshold,
            score_threshold: options.score_threshold,
            pre_nms_top_k: options.pre_nms_top_k,
            max_detections: options.max_detections,
        };
        batched_nms(
            &candidate_boxes,
            &candidate_scores,
            &candidate_class_ids,
            &nms_options,
        )?
    };

    let mut ordered_boxes = Vec::with_capacity(detections.len());
    let mut mapped_detections = Vec::with_capacity(detections.len());
    for detection in detections {
        let original_index = candidate_indices[detection.box_index];
        ordered_boxes.push(remapped_boxes[original_index]);
        mapped_detections.push(Detection {
            box_index: original_index,
            class_id: detection.class_id,
            score: detection.score,
        });
    }

    Ok(PostprocessResult {
        boxes: ordered_boxes,
        detections: mapped_detections,
    })
}

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
    validate_nms_inputs(boxes, scores, options)?;

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
    validate_thresholds(options)?;
    validate_boxes(boxes)?;

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

    validate_scores(class_scores)?;

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

    validate_soft_nms_options(options)?;
    validate_boxes(boxes)?;
    validate_scores(scores)?;

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

    validate_soft_nms_options(options)?;
    validate_boxes(boxes)?;
    validate_scores(scores)?;

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
    validate_soft_nms_options(options)?;
    validate_boxes(boxes)?;

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

    validate_scores(class_scores)?;

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

fn validate_nms_inputs(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    options: &NmsOptions,
) -> Result<(), BBoxError> {
    if boxes.len() != scores.len() {
        return Err(BBoxError::LengthMismatch {
            boxes: boxes.len(),
            scores: scores.len(),
        });
    }

    validate_thresholds(options)?;
    validate_boxes(boxes)?;
    validate_scores(scores)?;
    Ok(())
}

fn validate_postprocess_inputs(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    class_ids: &[usize],
    options: &PostprocessOptions,
) -> Result<(), BBoxError> {
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

    let nms_options = NmsOptions {
        iou_threshold: options.iou_threshold,
        score_threshold: options.score_threshold,
        pre_nms_top_k: options.pre_nms_top_k,
        max_detections: options.max_detections,
    };
    validate_thresholds(&nms_options)?;
    validate_boxes(boxes)?;
    validate_scores(scores)?;
    validate_min_size(options.min_width, options.min_height)?;
    Ok(())
}

fn remap_boxes_for_postprocess(
    boxes: &[BBoxXYXY],
    remap: BoxRemap,
) -> Result<(Vec<BBoxXYXY>, Option<(u32, u32)>), BBoxError> {
    match remap {
        BoxRemap::None => Ok((boxes.to_vec(), None)),
        BoxRemap::Current { width, height } => {
            validate_image_size(width, height)?;
            Ok((boxes.to_vec(), Some((width, height))))
        }
        BoxRemap::Resize {
            processed_width,
            processed_height,
            original_width,
            original_height,
        } => Ok((
            resize_boxes(
                boxes,
                processed_width,
                processed_height,
                original_width,
                original_height,
            )?,
            Some((original_width, original_height)),
        )),
        BoxRemap::Letterbox {
            processed_width,
            processed_height,
            original_width,
            original_height,
        } => Ok((
            unletterbox_boxes(
                boxes,
                original_width,
                original_height,
                processed_width,
                processed_height,
            )?,
            Some((original_width, original_height)),
        )),
    }
}

fn validate_thresholds(options: &NmsOptions) -> Result<(), BBoxError> {
    if !(0.0..=1.0).contains(&options.iou_threshold) {
        return Err(BBoxError::InvalidIouThreshold(options.iou_threshold));
    }

    if options.score_threshold.is_nan() {
        return Err(BBoxError::InvalidScoreThreshold(options.score_threshold));
    }

    Ok(())
}

fn validate_soft_nms_options(options: &SoftNmsOptions) -> Result<(), BBoxError> {
    if !(0.0..=1.0).contains(&options.iou_threshold) {
        return Err(BBoxError::InvalidIouThreshold(options.iou_threshold));
    }

    if options.score_threshold.is_nan() {
        return Err(BBoxError::InvalidScoreThreshold(options.score_threshold));
    }

    if !options.sigma.is_finite() || options.sigma <= 0.0 {
        return Err(BBoxError::InvalidSigma(options.sigma));
    }

    Ok(())
}

fn validate_boxes(boxes: &[BBoxXYXY]) -> Result<(), BBoxError> {
    for (index, bbox) in boxes.iter().enumerate() {
        if !bbox.is_finite() {
            return Err(BBoxError::NonFiniteBox { index });
        }
    }
    Ok(())
}

fn validate_image_size(width: u32, height: u32) -> Result<(), BBoxError> {
    if width == 0 || height == 0 {
        return Err(BBoxError::InvalidImageSize { width, height });
    }
    Ok(())
}

fn validate_area_bounds(min_area: Option<f32>, max_area: Option<f32>) -> Result<(), BBoxError> {
    if let Some(value) = min_area {
        if !value.is_finite() || value < 0.0 {
            return Err(BBoxError::InvalidMinArea(value));
        }
    }

    if let Some(value) = max_area {
        if !value.is_finite() || value < 0.0 {
            return Err(BBoxError::InvalidMaxArea(value));
        }
    }

    if let (Some(min_area), Some(max_area)) = (min_area, max_area) {
        if max_area < min_area {
            return Err(BBoxError::InvalidAreaRange { min_area, max_area });
        }
    }

    Ok(())
}

fn validate_min_size(min_width: f32, min_height: f32) -> Result<(), BBoxError> {
    let width_ok = min_width.is_finite() && min_width >= 0.0;
    let height_ok = min_height.is_finite() && min_height >= 0.0;

    if !width_ok || !height_ok {
        return Err(BBoxError::InvalidMinSize {
            min_width,
            min_height,
        });
    }

    Ok(())
}

fn validate_scores(scores: &[f32]) -> Result<(), BBoxError> {
    for (index, score) in scores.iter().copied().enumerate() {
        if !score.is_finite() {
            return Err(BBoxError::NonFiniteScore {
                index,
                value: score,
            });
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
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
}
