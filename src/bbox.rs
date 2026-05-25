/// Axis-aligned bounding box in `xyxy` format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBoxXYXY {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
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
}

/// Errors for box postprocessing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BBoxError {
    LengthMismatch { boxes: usize, scores: usize },
    InvalidIouThreshold(f32),
}

impl std::fmt::Display for BBoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { boxes, scores } => write!(
                f,
                "boxes and scores must have the same length, got {} boxes and {} scores",
                boxes, scores
            ),
            Self::InvalidIouThreshold(value) => write!(
                f,
                "iou_threshold must be in the inclusive range [0.0, 1.0], got {}",
                value
            ),
        }
    }
}

impl std::error::Error for BBoxError {}

/// Compute IoU between two `xyxy` boxes.
pub fn iou(a: BBoxXYXY, b: BBoxXYXY) -> f32 {
    a.iou(&b)
}

/// Run single-class non-maximum suppression.
///
/// Returns the kept indices in descending score order.
pub fn nms(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    iou_threshold: f32,
) -> Result<Vec<usize>, BBoxError> {
    if boxes.len() != scores.len() {
        return Err(BBoxError::LengthMismatch {
            boxes: boxes.len(),
            scores: scores.len(),
        });
    }

    if !(0.0..=1.0).contains(&iou_threshold) {
        return Err(BBoxError::InvalidIouThreshold(iou_threshold));
    }

    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut keep = Vec::with_capacity(order.len());

    'candidate: for idx in order {
        for &kept_idx in &keep {
            if boxes[idx].iou(&boxes[kept_idx]) > iou_threshold {
                continue 'candidate;
            }
        }
        keep.push(idx);
    }

    Ok(keep)
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
            BBoxError::LengthMismatch { boxes: 1, scores: 0 }
        );
        assert_eq!(
            nms(&boxes, &[0.5], 1.5).unwrap_err(),
            BBoxError::InvalidIouThreshold(1.5)
        );
    }
}
