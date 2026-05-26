use super::{BBoxError, BBoxXYWH, BBoxXYXY};

/// Compute IoU between two `xyxy` boxes.
pub fn iou(a: BBoxXYXY, b: BBoxXYXY) -> f32 {
    a.iou(&b)
}

/// Convert boxes from `xyxy` into `xywh` format.
pub fn xyxy_to_xywh(boxes: &[BBoxXYXY]) -> Result<Vec<BBoxXYWH>, BBoxError> {
    super::validate_boxes(boxes)?;
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
