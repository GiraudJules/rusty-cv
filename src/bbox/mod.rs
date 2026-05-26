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

type ClipBounds = Option<(u32, u32)>;
type RemappedBoxes = (Vec<BBoxXYXY>, ClipBounds);

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

mod filtering;
mod geometry;
mod nms;
mod postprocess;
mod remap;

pub use filtering::{
    clip_and_filter_boxes, clip_boxes, filter_boxes_by_area, filter_boxes_by_min_size,
    filter_boxes_by_score,
};
pub use geometry::{iou, xywh_to_xyxy, xyxy_to_xywh};
pub use nms::{
    batched_nms, batched_soft_nms, multiclass_nms, multiclass_soft_nms, nms, nms_with_options,
    soft_nms,
};
pub use postprocess::postprocess_detections;
pub use remap::{letterbox_boxes, resize_boxes, unletterbox_boxes};

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

#[cfg(test)]
mod tests;
