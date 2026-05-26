use crate::bbox::BBoxXYXY;
use crate::letterbox::{self, LetterboxError, LetterboxInfo};

/// Result of resizing a mask to exact dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResizeMaskResult {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Result of letterboxing a mask.
#[derive(Debug, Clone, PartialEq)]
pub struct LetterboxMaskResult {
    pub data: Vec<u8>,
    pub info: LetterboxInfo,
}

/// Errors for segmentation mask operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskError {
    InvalidDimensions { width: u32, height: u32 },
    InvalidDataLength { expected: usize, actual: usize },
    InvalidThreshold(f32),
    Letterbox(LetterboxError),
}

impl std::fmt::Display for MaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => write!(
                f,
                "mask width and height must be greater than zero, got {}x{}",
                width, height
            ),
            Self::InvalidDataLength { expected, actual } => write!(
                f,
                "mask data length does not match shape, expected {} values but got {}",
                expected, actual
            ),
            Self::InvalidThreshold(value) => {
                write!(f, "threshold must be finite, got {}", value)
            }
            Self::Letterbox(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for MaskError {}

/// Resize a mask using nearest-neighbor sampling.
pub fn resize_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<ResizeMaskResult, MaskError> {
    validate_mask(mask, width, height)?;
    validate_dimensions(target_width, target_height)?;

    let width_usize = width as usize;
    let height_usize = height as usize;
    let target_width_usize = target_width as usize;
    let target_height_usize = target_height as usize;
    let mut output = vec![0u8; target_width_usize * target_height_usize];

    for y in 0..target_height_usize {
        let source_y = y * height_usize / target_height_usize;
        for x in 0..target_width_usize {
            let source_x = x * width_usize / target_width_usize;
            output[y * target_width_usize + x] = mask[source_y * width_usize + source_x];
        }
    }

    Ok(ResizeMaskResult {
        data: output,
        width: target_width,
        height: target_height,
    })
}

/// Letterbox a mask into the target dimensions.
pub fn letterbox_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    fill: u8,
) -> Result<LetterboxMaskResult, MaskError> {
    validate_mask(mask, width, height)?;
    let info = letterbox::compute_letterbox(width, height, target_width, target_height)
        .map_err(MaskError::Letterbox)?;
    let resized = resize_mask(mask, width, height, info.resized_width, info.resized_height)?;
    let mut canvas = vec![fill; target_width as usize * target_height as usize];

    for y in 0..info.resized_height as usize {
        let dest_y = y + info.padding.top as usize;
        let dest_row_start = dest_y * target_width as usize + info.padding.left as usize;
        let src_row_start = y * info.resized_width as usize;
        let src_row_end = src_row_start + info.resized_width as usize;
        canvas[dest_row_start..dest_row_start + info.resized_width as usize]
            .copy_from_slice(&resized.data[src_row_start..src_row_end]);
    }

    Ok(LetterboxMaskResult { data: canvas, info })
}

/// Remove letterbox padding and resize the mask back to the original dimensions.
pub fn unletterbox_mask(
    mask: &[u8],
    target_width: u32,
    target_height: u32,
    original_width: u32,
    original_height: u32,
) -> Result<ResizeMaskResult, MaskError> {
    validate_mask(mask, target_width, target_height)?;
    let info =
        letterbox::compute_letterbox(original_width, original_height, target_width, target_height)
            .map_err(MaskError::Letterbox)?;

    let cropped_width = info.resized_width as usize;
    let cropped_height = info.resized_height as usize;
    let mut cropped = vec![0u8; cropped_width * cropped_height];

    for y in 0..cropped_height {
        let src_y = y + info.padding.top as usize;
        let src_row_start = src_y * target_width as usize + info.padding.left as usize;
        let src_row_end = src_row_start + cropped_width;
        let dst_row_start = y * cropped_width;
        cropped[dst_row_start..dst_row_start + cropped_width]
            .copy_from_slice(&mask[src_row_start..src_row_end]);
    }

    resize_mask(
        &cropped,
        info.resized_width,
        info.resized_height,
        original_width,
        original_height,
    )
}

/// Threshold a float mask into a binary `u8` mask with values `{0, 255}`.
pub fn threshold_mask(
    mask: &[f32],
    width: u32,
    height: u32,
    threshold: f32,
) -> Result<Vec<u8>, MaskError> {
    validate_mask_length(mask.len(), width, height)?;
    if !threshold.is_finite() {
        return Err(MaskError::InvalidThreshold(threshold));
    }

    Ok(mask
        .iter()
        .map(|value| if *value >= threshold { 255 } else { 0 })
        .collect())
}

/// Compute one bounding box covering all non-zero pixels in a mask.
pub fn mask_to_box(mask: &[u8], width: u32, height: u32) -> Result<Option<BBoxXYXY>, MaskError> {
    validate_mask(mask, width, height)?;

    let width_usize = width as usize;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..height as usize {
        for x in 0..width_usize {
            if mask[y * width_usize + x] != 0 {
                found = true;
                min_x = min_x.min(x as u32);
                min_y = min_y.min(y as u32);
                max_x = max_x.max(x as u32 + 1);
                max_y = max_y.max(y as u32 + 1);
            }
        }
    }

    if !found {
        return Ok(None);
    }

    Ok(Some(BBoxXYXY {
        x1: min_x as f32,
        y1: min_y as f32,
        x2: max_x as f32,
        y2: max_y as f32,
    }))
}

fn validate_mask(mask: &[u8], width: u32, height: u32) -> Result<(), MaskError> {
    validate_dimensions(width, height)?;
    validate_mask_length(mask.len(), width, height)
}

fn validate_mask_length(actual_len: usize, width: u32, height: u32) -> Result<(), MaskError> {
    let expected = width as usize * height as usize;
    if actual_len != expected {
        return Err(MaskError::InvalidDataLength {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), MaskError> {
    if width == 0 || height == 0 {
        return Err(MaskError::InvalidDimensions { width, height });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resizes_mask_with_nearest_neighbor() {
        let mask = vec![0u8, 255, 255, 0];
        let result = resize_mask(&mask, 2, 2, 4, 2).unwrap();

        assert_eq!(result.width, 4);
        assert_eq!(result.height, 2);
        assert_eq!(result.data, vec![0, 0, 255, 255, 255, 255, 0, 0]);
    }

    #[test]
    fn letterboxes_and_unletterboxes_mask() {
        let mask = vec![0u8, 255];
        let letterboxed = letterbox_mask(&mask, 2, 1, 4, 4, 7).unwrap();

        assert_eq!(letterboxed.info.resized_width, 4);
        assert_eq!(letterboxed.info.padding.top, 1);
        assert_eq!(letterboxed.data[0], 7);

        let restored = unletterbox_mask(&letterboxed.data, 4, 4, 2, 1).unwrap();
        assert_eq!(restored.data, mask);
    }

    #[test]
    fn thresholds_mask() {
        let mask = vec![0.1f32, 0.6, 0.5, 0.2];
        let thresholded = threshold_mask(&mask, 2, 2, 0.5).unwrap();
        assert_eq!(thresholded, vec![0, 255, 255, 0]);
    }

    #[test]
    fn extracts_mask_box() {
        let mask = vec![0u8, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0];
        let bbox = mask_to_box(&mask, 4, 4).unwrap().unwrap();
        assert_eq!(
            bbox,
            BBoxXYXY {
                x1: 1.0,
                y1: 1.0,
                x2: 3.0,
                y2: 3.0,
            }
        );
    }

    #[test]
    fn rejects_invalid_masks() {
        assert_eq!(
            resize_mask(&[0u8], 0, 1, 1, 1).unwrap_err(),
            MaskError::InvalidDimensions {
                width: 0,
                height: 1
            }
        );
        assert!(matches!(
            threshold_mask(&[0.0, 0.0, 0.0, 0.0], 2, 2, f32::NAN).unwrap_err(),
            MaskError::InvalidThreshold(value) if value.is_nan()
        ));
        assert_eq!(
            mask_to_box(&[0u8], 2, 2).unwrap_err(),
            MaskError::InvalidDataLength {
                expected: 4,
                actual: 1,
            }
        );
    }
}
