use image::imageops;
use image::{DynamicImage, RgbImage};

/// Metadata describing a crop operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropInfo {
    pub original_width: u32,
    pub original_height: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Result of a crop operation.
#[derive(Debug, Clone, PartialEq)]
pub struct CropResult {
    pub image: RgbImage,
    pub info: CropInfo,
}

/// Errors for crop operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CropError {
    ZeroSizedImage,
    ZeroSizedCrop,
    CropOutOfBounds,
}

impl std::fmt::Display for CropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSizedImage => {
                f.write_str("source image width and height must be greater than zero")
            }
            Self::ZeroSizedCrop => f.write_str("crop width and height must be greater than zero"),
            Self::CropOutOfBounds => {
                f.write_str("crop rectangle must fit inside the source image bounds")
            }
        }
    }
}

impl std::error::Error for CropError {}

/// Crop an image to the exact requested rectangle.
pub fn crop_image(
    image: &DynamicImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<CropResult, CropError> {
    let rgb = image.to_rgb8();

    if rgb.width() == 0 || rgb.height() == 0 {
        return Err(CropError::ZeroSizedImage);
    }

    if width == 0 || height == 0 {
        return Err(CropError::ZeroSizedCrop);
    }

    let fits_horizontally = x
        .checked_add(width)
        .is_some_and(|right| right <= rgb.width());
    let fits_vertically = y
        .checked_add(height)
        .is_some_and(|bottom| bottom <= rgb.height());

    if !fits_horizontally || !fits_vertically {
        return Err(CropError::CropOutOfBounds);
    }

    let cropped = imageops::crop_imm(&rgb, x, y, width, height).to_image();

    Ok(CropResult {
        image: cropped,
        info: CropInfo {
            original_width: rgb.width(),
            original_height: rgb.height(),
            x,
            y,
            width,
            height,
        },
    })
}

/// Crop the centered region of an image to the requested size.
pub fn center_crop_image(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> Result<CropResult, CropError> {
    let rgb = image.to_rgb8();

    if rgb.width() == 0 || rgb.height() == 0 {
        return Err(CropError::ZeroSizedImage);
    }

    if width == 0 || height == 0 {
        return Err(CropError::ZeroSizedCrop);
    }

    if width > rgb.width() || height > rgb.height() {
        return Err(CropError::CropOutOfBounds);
    }

    let x = (rgb.width() - width) / 2;
    let y = (rgb.height() - height) / 2;

    crop_image(&DynamicImage::ImageRgb8(rgb), x, y, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn gradient_rgb(width: u32, height: u32) -> DynamicImage {
        let image = ImageBuffer::from_fn(width, height, |x, y| Rgb([x as u8, y as u8, 0]));
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn crops_exact_region() {
        let image = gradient_rgb(5, 4);
        let result = crop_image(&image, 1, 1, 3, 2).unwrap();

        assert_eq!(result.image.width(), 3);
        assert_eq!(result.image.height(), 2);
        assert_eq!(result.image.get_pixel(0, 0).0, [1, 1, 0]);
        assert_eq!(result.image.get_pixel(2, 1).0, [3, 2, 0]);
    }

    #[test]
    fn center_crop_uses_middle_region() {
        let image = gradient_rgb(6, 4);
        let result = center_crop_image(&image, 2, 2).unwrap();

        assert_eq!(result.info.x, 2);
        assert_eq!(result.info.y, 1);
        assert_eq!(result.image.get_pixel(0, 0).0, [2, 1, 0]);
        assert_eq!(result.image.get_pixel(1, 1).0, [3, 2, 0]);
    }

    #[test]
    fn rejects_out_of_bounds_crop() {
        let image = gradient_rgb(4, 4);
        assert_eq!(
            crop_image(&image, 3, 3, 2, 2).unwrap_err(),
            CropError::CropOutOfBounds
        );
    }

    #[test]
    fn rejects_zero_sized_crop() {
        let image = gradient_rgb(4, 4);
        assert_eq!(
            crop_image(&image, 0, 0, 0, 2).unwrap_err(),
            CropError::ZeroSizedCrop
        );
    }
}
