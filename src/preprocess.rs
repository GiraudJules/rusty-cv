use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};

use crate::letterbox::{self, LetterboxError, LetterboxInfo};
use crate::resize::{self, ResizeError, ResizeInfo};

/// Geometry mode used during preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessMode {
    Resize,
    Letterbox { fill: [u8; 3] },
}

/// Output memory layout used for normalized tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessLayout {
    Hwc,
    Chw,
}

/// Geometry metadata recorded during preprocessing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreprocessGeometry {
    Resize(ResizeInfo),
    Letterbox(LetterboxInfo),
}

/// Metadata describing a fused preprocessing operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessInfo {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub scale_to_unit: bool,
    pub layout: PreprocessLayout,
    pub geometry: PreprocessGeometry,
}

/// Result of a fused preprocessing operation.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessResult {
    pub data: Vec<f32>,
    pub info: PreprocessInfo,
}

/// Errors for fused preprocessing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessError {
    Resize(ResizeError),
    Letterbox(LetterboxError),
    NonPositiveStd { channel: usize, value: f32 },
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resize(err) => err.fmt(f),
            Self::Letterbox(err) => err.fmt(f),
            Self::NonPositiveStd { channel, value } => write!(
                f,
                "standard deviation for channel {} must be greater than zero, got {}",
                channel, value
            ),
        }
    }
}

impl std::error::Error for PreprocessError {}

fn normalize_rgb_image(
    image: &RgbImage,
    mean: [f32; 3],
    std: [f32; 3],
    scale_to_unit: bool,
    layout: PreprocessLayout,
) -> Result<Vec<f32>, PreprocessError> {
    for (channel, value) in std.into_iter().enumerate() {
        if value <= 0.0 {
            return Err(PreprocessError::NonPositiveStd { channel, value });
        }
    }

    let width = image.width() as usize;
    let height = image.height() as usize;
    let plane = width * height;
    let scale = if scale_to_unit { 1.0 / 255.0 } else { 1.0 };
    let mut data = vec![0.0; plane * 3];

    for (index, pixel) in image.pixels().enumerate() {
        match layout {
            PreprocessLayout::Hwc => {
                let base = index * 3;
                for channel in 0..3 {
                    let value = pixel.0[channel] as f32 * scale;
                    data[base + channel] = (value - mean[channel]) / std[channel];
                }
            }
            PreprocessLayout::Chw => {
                for channel in 0..3 {
                    let value = pixel.0[channel] as f32 * scale;
                    data[channel * plane + index] = (value - mean[channel]) / std[channel];
                }
            }
        }
    }

    Ok(data)
}

/// Resize or letterbox an image and normalize it into an `f32` tensor buffer.
///
/// This is intended for inference preprocessing paths where resizing,
/// normalization, and tensor layout conversion should happen in one Rust call.
#[allow(clippy::too_many_arguments)]
pub fn preprocess_image(
    image: &DynamicImage,
    target_width: u32,
    target_height: u32,
    mode: PreprocessMode,
    filter: FilterType,
    mean: [f32; 3],
    std: [f32; 3],
    scale_to_unit: bool,
    layout: PreprocessLayout,
) -> Result<PreprocessResult, PreprocessError> {
    let (processed, geometry) = match mode {
        PreprocessMode::Resize => {
            let result = resize::resize_image(image, target_width, target_height, filter)
                .map_err(PreprocessError::Resize)?;
            (result.image, PreprocessGeometry::Resize(result.info))
        }
        PreprocessMode::Letterbox { fill } => {
            let result =
                letterbox::letterbox_image(image, target_width, target_height, fill, filter)
                    .map_err(PreprocessError::Letterbox)?;
            (result.image, PreprocessGeometry::Letterbox(result.info))
        }
    };

    let width = processed.width();
    let height = processed.height();
    let data = normalize_rgb_image(&processed, mean, std, scale_to_unit, layout)?;

    Ok(PreprocessResult {
        data,
        info: PreprocessInfo {
            width,
            height,
            channels: 3,
            scale_to_unit,
            layout,
            geometry,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn image_from_pixels(width: u32, height: u32, pixels: Vec<[u8; 3]>) -> DynamicImage {
        let raw = pixels.into_iter().flatten().collect::<Vec<u8>>();
        let image = RgbImage::from_vec(width, height, raw).unwrap();
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn preprocesses_into_chw_tensor() {
        let image = image_from_pixels(2, 1, vec![[255, 0, 0], [0, 255, 0]]);
        let result = preprocess_image(
            &image,
            2,
            1,
            PreprocessMode::Resize,
            FilterType::Nearest,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
            PreprocessLayout::Chw,
        )
        .unwrap();

        assert_eq!(result.info.width, 2);
        assert_eq!(result.info.height, 1);
        assert_eq!(result.info.layout, PreprocessLayout::Chw);
        assert_eq!(result.data, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            result.info.geometry,
            PreprocessGeometry::Resize(ResizeInfo {
                original_width: 2,
                original_height: 1,
                target_width: 2,
                target_height: 1,
            })
        );
    }

    #[test]
    fn preprocesses_letterbox_into_hwc_tensor() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 1, Rgb([10, 20, 30])));
        let result = preprocess_image(
            &image,
            2,
            3,
            PreprocessMode::Letterbox {
                fill: [114, 114, 114],
            },
            FilterType::Nearest,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            false,
            PreprocessLayout::Hwc,
        )
        .unwrap();

        assert_eq!(result.info.width, 2);
        assert_eq!(result.info.height, 3);
        assert_eq!(result.info.layout, PreprocessLayout::Hwc);
        assert_eq!(&result.data[0..3], &[114.0, 114.0, 114.0]);
        assert_eq!(&result.data[6..9], &[10.0, 20.0, 30.0]);
        assert_eq!(
            result.info.geometry,
            PreprocessGeometry::Letterbox(LetterboxInfo {
                original_width: 2,
                original_height: 1,
                target_width: 2,
                target_height: 3,
                resized_width: 2,
                resized_height: 1,
                scale: 1.0,
                padding: crate::letterbox::Padding {
                    top: 1,
                    bottom: 1,
                    left: 0,
                    right: 0,
                },
            })
        );
    }

    #[test]
    fn rejects_non_positive_std() {
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1, 1, Rgb([0, 0, 0])));
        let err = preprocess_image(
            &image,
            1,
            1,
            PreprocessMode::Resize,
            FilterType::Nearest,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            true,
            PreprocessLayout::Hwc,
        )
        .unwrap_err();

        assert_eq!(
            err,
            PreprocessError::NonPositiveStd {
                channel: 1,
                value: 0.0,
            }
        );
    }
}
