/// Errors returned by tensor layout conversion helpers.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutError {
    InvalidDimension { name: &'static str, value: u32 },
    InvalidChannels(u32),
    InvalidDataLength { expected: usize, actual: usize },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension { name, value } => {
                write!(f, "{name} must be greater than zero, got {value}")
            }
            Self::InvalidChannels(value) => {
                write!(f, "channels must be greater than zero, got {}", value)
            }
            Self::InvalidDataLength { expected, actual } => write!(
                f,
                "tensor data length does not match shape, expected {} values but got {}",
                expected, actual
            ),
        }
    }
}

impl std::error::Error for LayoutError {}

/// Convert a HWC tensor into CHW layout.
pub fn hwc_to_chw<T: Copy>(
    data: &[T],
    height: u32,
    width: u32,
    channels: u32,
) -> Result<Vec<T>, LayoutError> {
    validate_3d_shape(data.len(), height, width, channels)?;
    let height = height as usize;
    let width = width as usize;
    let channels = channels as usize;

    let mut output = Vec::with_capacity(data.len());
    for channel in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let input_index = ((y * width) + x) * channels + channel;
                output.push(data[input_index]);
            }
        }
    }
    Ok(output)
}

/// Convert a CHW tensor into HWC layout.
pub fn chw_to_hwc<T: Copy>(
    data: &[T],
    channels: u32,
    height: u32,
    width: u32,
) -> Result<Vec<T>, LayoutError> {
    validate_3d_shape(data.len(), height, width, channels)?;
    let height = height as usize;
    let width = width as usize;
    let channels = channels as usize;

    let mut output = Vec::with_capacity(data.len());
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                let input_index = ((channel * height) + y) * width + x;
                output.push(data[input_index]);
            }
        }
    }
    Ok(output)
}

/// Convert a NHWC tensor into NCHW layout.
pub fn nhwc_to_nchw<T: Copy>(
    data: &[T],
    batch: u32,
    height: u32,
    width: u32,
    channels: u32,
) -> Result<Vec<T>, LayoutError> {
    validate_4d_shape(data.len(), batch, height, width, channels)?;
    let batch = batch as usize;
    let height = height as usize;
    let width = width as usize;
    let channels = channels as usize;

    let mut output = Vec::with_capacity(data.len());
    for n in 0..batch {
        for channel in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let input_index = (((n * height) + y) * width + x) * channels + channel;
                    output.push(data[input_index]);
                }
            }
        }
    }
    Ok(output)
}

/// Convert a NCHW tensor into NHWC layout.
pub fn nchw_to_nhwc<T: Copy>(
    data: &[T],
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
) -> Result<Vec<T>, LayoutError> {
    validate_4d_shape(data.len(), batch, height, width, channels)?;
    let batch = batch as usize;
    let height = height as usize;
    let width = width as usize;
    let channels = channels as usize;

    let mut output = Vec::with_capacity(data.len());
    for n in 0..batch {
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let input_index = (((n * channels) + channel) * height + y) * width + x;
                    output.push(data[input_index]);
                }
            }
        }
    }
    Ok(output)
}

/// Swap the channel order of an RGB HWC tensor into BGR.
pub fn rgb_to_bgr<T: Copy>(data: &[T], height: u32, width: u32) -> Result<Vec<T>, LayoutError> {
    validate_3d_shape(data.len(), height, width, 3)?;
    let height = height as usize;
    let width = width as usize;

    let mut output = Vec::with_capacity(data.len());
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width) + x) * 3;
            output.push(data[offset + 2]);
            output.push(data[offset + 1]);
            output.push(data[offset]);
        }
    }
    Ok(output)
}

fn validate_3d_shape(
    actual_len: usize,
    height: u32,
    width: u32,
    channels: u32,
) -> Result<(), LayoutError> {
    validate_positive_dimension("height", height)?;
    validate_positive_dimension("width", width)?;
    if channels == 0 {
        return Err(LayoutError::InvalidChannels(channels));
    }

    let expected = height as usize * width as usize * channels as usize;
    if actual_len != expected {
        return Err(LayoutError::InvalidDataLength {
            expected,
            actual: actual_len,
        });
    }

    Ok(())
}

fn validate_4d_shape(
    actual_len: usize,
    batch: u32,
    height: u32,
    width: u32,
    channels: u32,
) -> Result<(), LayoutError> {
    validate_positive_dimension("batch", batch)?;
    validate_3d_shape(actual_len / batch as usize, height, width, channels)?;

    let expected = batch as usize * height as usize * width as usize * channels as usize;
    if actual_len != expected {
        return Err(LayoutError::InvalidDataLength {
            expected,
            actual: actual_len,
        });
    }

    Ok(())
}

fn validate_positive_dimension(name: &'static str, value: u32) -> Result<(), LayoutError> {
    if value == 0 {
        return Err(LayoutError::InvalidDimension { name, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_hwc_to_chw_and_back() {
        let input = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let chw = hwc_to_chw(&input, 2, 2, 3).unwrap();
        assert_eq!(chw, vec![1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]);

        let roundtrip = chw_to_hwc(&chw, 3, 2, 2).unwrap();
        assert_eq!(roundtrip, input);
    }

    #[test]
    fn converts_nhwc_to_nchw_and_back() {
        let input = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let nchw = nhwc_to_nchw(&input, 1, 2, 2, 2).unwrap();
        assert_eq!(nchw, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);

        let roundtrip = nchw_to_nhwc(&nchw, 1, 2, 2, 2).unwrap();
        assert_eq!(roundtrip, input);
    }

    #[test]
    fn swaps_rgb_to_bgr() {
        let input = vec![255u8, 0, 10, 0, 128, 255];
        let output = rgb_to_bgr(&input, 1, 2).unwrap();
        assert_eq!(output, vec![10, 0, 255, 255, 128, 0]);
    }

    #[test]
    fn rejects_invalid_shapes() {
        assert_eq!(
            hwc_to_chw::<u8>(&[1, 2, 3], 0, 1, 3).unwrap_err(),
            LayoutError::InvalidDimension {
                name: "height",
                value: 0,
            }
        );
        assert_eq!(
            hwc_to_chw::<u8>(&[1, 2, 3], 1, 1, 0).unwrap_err(),
            LayoutError::InvalidChannels(0)
        );
        assert_eq!(
            chw_to_hwc::<u8>(&[1, 2, 3], 3, 1, 2).unwrap_err(),
            LayoutError::InvalidDataLength {
                expected: 6,
                actual: 3,
            }
        );
    }
}
