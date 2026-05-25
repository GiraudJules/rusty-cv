use image::imageops::{self, FilterType};
use image::{DynamicImage, RgbImage};

/// Metadata describing a direct resize operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeInfo {
    pub original_width: u32,
    pub original_height: u32,
    pub target_width: u32,
    pub target_height: u32,
}

/// Result of a resize operation.
#[derive(Debug, Clone, PartialEq)]
pub struct ResizeResult {
    pub image: RgbImage,
    pub info: ResizeInfo,
}

/// Errors for direct resize operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResizeError {
    ZeroSizedImage,
    ZeroSizedTarget,
}

impl std::fmt::Display for ResizeError {
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

impl std::error::Error for ResizeError {}

/// Resize an image to the exact requested width and height.
///
/// This operation does not preserve aspect ratio. For aspect-ratio-preserving
/// resize plus padding, use [`crate::letterbox_image`].
pub fn resize_image(
    image: &DynamicImage,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> Result<ResizeResult, ResizeError> {
    let rgb = image.to_rgb8();

    if rgb.width() == 0 || rgb.height() == 0 {
        return Err(ResizeError::ZeroSizedImage);
    }

    if target_width == 0 || target_height == 0 {
        return Err(ResizeError::ZeroSizedTarget);
    }

    let resized = imageops::resize(&rgb, target_width, target_height, filter);

    Ok(ResizeResult {
        image: resized,
        info: ResizeInfo {
            original_width: rgb.width(),
            original_height: rgb.height(),
            target_width,
            target_height,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_rgb(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        let image = ImageBuffer::from_pixel(width, height, Rgb(color));
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn resizes_to_exact_dimensions() {
        let image = solid_rgb(4, 2, [12, 34, 56]);
        let result = resize_image(&image, 8, 8, FilterType::Nearest).unwrap();

        assert_eq!(result.image.width(), 8);
        assert_eq!(result.image.height(), 8);
        assert_eq!(result.image.get_pixel(0, 0).0, [12, 34, 56]);
        assert_eq!(result.image.get_pixel(7, 7).0, [12, 34, 56]);
    }

    #[test]
    fn reports_resize_metadata() {
        let image = solid_rgb(10, 20, [0, 0, 0]);
        let result = resize_image(&image, 32, 16, FilterType::Triangle).unwrap();

        assert_eq!(
            result.info,
            ResizeInfo {
                original_width: 10,
                original_height: 20,
                target_width: 32,
                target_height: 16,
            }
        );
    }

    #[test]
    fn rejects_zero_sized_target() {
        let image = solid_rgb(4, 4, [0, 0, 0]);
        assert_eq!(
            resize_image(&image, 0, 8, FilterType::Nearest).unwrap_err(),
            ResizeError::ZeroSizedTarget
        );
    }
}
