use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

/// Padding applied around a resized image during letterboxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
}

/// Geometry metadata describing a letterbox transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxInfo {
    pub original_width: u32,
    pub original_height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub scale: f32,
    pub padding: Padding,
}

/// Result of a letterbox image transform.
#[derive(Debug, Clone, PartialEq)]
pub struct LetterboxResult {
    pub image: RgbImage,
    pub info: LetterboxInfo,
}

/// Errors for letterbox operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LetterboxError {
    ZeroSizedImage,
    ZeroSizedTarget,
}

impl std::fmt::Display for LetterboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSizedImage => {
                f.write_str("source image width and height must be greater than zero")
            }
            Self::ZeroSizedTarget => {
                f.write_str("target width and height must be greater than zero")
            }
        }
    }
}

impl std::error::Error for LetterboxError {}

/// Compute the geometry of a letterbox operation without resizing pixels.
pub fn compute_letterbox(
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<LetterboxInfo, LetterboxError> {
    if original_width == 0 || original_height == 0 {
        return Err(LetterboxError::ZeroSizedImage);
    }

    if target_width == 0 || target_height == 0 {
        return Err(LetterboxError::ZeroSizedTarget);
    }

    let scale = f32::min(
        target_width as f32 / original_width as f32,
        target_height as f32 / original_height as f32,
    );

    let resized_width = ((original_width as f32 * scale).round() as u32)
        .max(1)
        .min(target_width);
    let resized_height = ((original_height as f32 * scale).round() as u32)
        .max(1)
        .min(target_height);

    let pad_x = target_width.saturating_sub(resized_width);
    let pad_y = target_height.saturating_sub(resized_height);

    let padding = Padding {
        left: pad_x / 2,
        right: pad_x - (pad_x / 2),
        top: pad_y / 2,
        bottom: pad_y - (pad_y / 2),
    };

    Ok(LetterboxInfo {
        original_width,
        original_height,
        target_width,
        target_height,
        resized_width,
        resized_height,
        scale,
        padding,
    })
}

/// Resize an image to fit inside the target frame and pad the remaining area.
///
/// The original aspect ratio is preserved, and the image is centered inside
/// the output canvas using the provided fill color.
pub fn letterbox_image(
    image: &DynamicImage,
    target_width: u32,
    target_height: u32,
    fill: [u8; 3],
    filter: FilterType,
) -> Result<LetterboxResult, LetterboxError> {
    let rgb = image.to_rgb8();
    let info = compute_letterbox(rgb.width(), rgb.height(), target_width, target_height)?;

    let resized = imageops::resize(&rgb, info.resized_width, info.resized_height, filter);
    let mut canvas: RgbImage = ImageBuffer::from_pixel(target_width, target_height, Rgb(fill));

    imageops::replace(
        &mut canvas,
        &resized,
        i64::from(info.padding.left),
        i64::from(info.padding.top),
    );

    Ok(LetterboxResult {
        image: canvas,
        info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        let image = ImageBuffer::from_pixel(width, height, Rgb(color));
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn computes_expected_padding_for_wide_image() {
        let info = compute_letterbox(4, 2, 8, 8).unwrap();

        assert_eq!(info.resized_width, 8);
        assert_eq!(info.resized_height, 4);
        assert_eq!(
            info.padding,
            Padding {
                top: 2,
                bottom: 2,
                left: 0,
                right: 0,
            }
        );
    }

    #[test]
    fn computes_expected_padding_for_tall_image() {
        let info = compute_letterbox(3, 7, 10, 10).unwrap();

        assert_eq!(info.resized_width, 4);
        assert_eq!(info.resized_height, 10);
        assert_eq!(
            info.padding,
            Padding {
                top: 0,
                bottom: 0,
                left: 3,
                right: 3,
            }
        );
    }

    #[test]
    fn paints_padding_with_fill_color() {
        let image = solid_rgb(4, 2, [255, 0, 0]);
        let result = letterbox_image(&image, 8, 8, [114, 114, 114], FilterType::Nearest).unwrap();

        assert_eq!(result.image.width(), 8);
        assert_eq!(result.image.height(), 8);
        assert_eq!(result.image.get_pixel(0, 0).0, [114, 114, 114]);
        assert_eq!(result.image.get_pixel(4, 3).0, [255, 0, 0]);
        assert_eq!(result.image.get_pixel(7, 7).0, [114, 114, 114]);
    }

    #[test]
    fn rejects_zero_sized_inputs() {
        assert_eq!(
            compute_letterbox(0, 10, 20, 20).unwrap_err(),
            LetterboxError::ZeroSizedImage
        );
        assert_eq!(
            compute_letterbox(10, 10, 0, 20).unwrap_err(),
            LetterboxError::ZeroSizedTarget
        );
    }
}
