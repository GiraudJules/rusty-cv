use image::DynamicImage;

/// Metadata describing a normalization operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizeInfo {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub scale_to_unit: bool,
}

/// Result of a normalization operation.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeResult {
    pub data: Vec<f32>,
    pub info: NormalizeInfo,
}

/// Errors for normalization operations.
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizeError {
    ZeroSizedImage,
    NonPositiveStd { channel: usize, value: f32 },
}

impl std::fmt::Display for NormalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSizedImage => {
                f.write_str("source image width and height must be greater than zero")
            }
            Self::NonPositiveStd { channel, value } => write!(
                f,
                "standard deviation for channel {} must be greater than zero, got {}",
                channel, value
            ),
        }
    }
}

impl std::error::Error for NormalizeError {}

/// Normalize an RGB image into a contiguous HWC `f32` buffer.
///
/// If `scale_to_unit` is `true`, pixel values are first scaled from `0..=255`
/// into `0.0..=1.0` before subtracting `mean` and dividing by `std`.
pub fn normalize_image(
    image: &DynamicImage,
    mean: [f32; 3],
    std: [f32; 3],
    scale_to_unit: bool,
) -> Result<NormalizeResult, NormalizeError> {
    let rgb = image.to_rgb8();

    if rgb.width() == 0 || rgb.height() == 0 {
        return Err(NormalizeError::ZeroSizedImage);
    }

    for (channel, value) in std.into_iter().enumerate() {
        if value <= 0.0 {
            return Err(NormalizeError::NonPositiveStd { channel, value });
        }
    }

    let scale = if scale_to_unit { 1.0 / 255.0 } else { 1.0 };
    let mut data = Vec::with_capacity((rgb.width() * rgb.height() * 3) as usize);

    for pixel in rgb.pixels() {
        for channel in 0..3 {
            let value = pixel.0[channel] as f32 * scale;
            data.push((value - mean[channel]) / std[channel]);
        }
    }

    Ok(NormalizeResult {
        data,
        info: NormalizeInfo {
            width: rgb.width(),
            height: rgb.height(),
            channels: 3,
            scale_to_unit,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgb};

    fn solid_rgb(width: u32, height: u32, color: [u8; 3]) -> DynamicImage {
        let image = ImageBuffer::from_pixel(width, height, Rgb(color));
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn normalizes_into_hwc_f32_buffer() {
        let image = solid_rgb(1, 1, [255, 127, 0]);
        let result = normalize_image(&image, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], true).unwrap();

        assert_eq!(result.info.width, 1);
        assert_eq!(result.info.height, 1);
        assert_eq!(result.info.channels, 3);
        assert!((result.data[0] - 1.0).abs() < 1e-6);
        assert!((result.data[1] - (127.0 / 255.0)).abs() < 1e-6);
        assert_eq!(result.data[2], 0.0);
    }

    #[test]
    fn applies_mean_and_std() {
        let image = solid_rgb(1, 1, [10, 20, 30]);
        let result = normalize_image(&image, [5.0, 10.0, 15.0], [5.0, 10.0, 15.0], false).unwrap();

        assert_eq!(result.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_non_positive_std() {
        let image = solid_rgb(1, 1, [0, 0, 0]);
        assert_eq!(
            normalize_image(&image, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], true).unwrap_err(),
            NormalizeError::NonPositiveStd {
                channel: 1,
                value: 0.0
            }
        );
    }
}
