use super::{BBoxError, BBoxXYXY, BoxRemap, RemappedBoxes};

use crate::letterbox::compute_letterbox;

/// Map boxes from one image size to another with direct resize scaling.
pub fn resize_boxes(
    boxes: &[BBoxXYXY],
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<BBoxXYXY>, BBoxError> {
    super::validate_boxes(boxes)?;
    super::validate_image_size(original_width, original_height)?;
    super::validate_image_size(target_width, target_height)?;

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
    super::validate_boxes(boxes)?;
    super::validate_image_size(original_width, original_height)?;
    super::validate_image_size(target_width, target_height)?;
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
    super::validate_boxes(boxes)?;
    super::validate_image_size(original_width, original_height)?;
    super::validate_image_size(target_width, target_height)?;
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

pub(super) fn remap_boxes_for_postprocess(
    boxes: &[BBoxXYXY],
    remap: BoxRemap,
) -> Result<RemappedBoxes, BBoxError> {
    match remap {
        BoxRemap::None => Ok((boxes.to_vec(), None)),
        BoxRemap::Current { width, height } => {
            super::validate_image_size(width, height)?;
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
