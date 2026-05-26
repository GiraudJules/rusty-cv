use image::DynamicImage;
use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};
use pyo3::wrap_pyfunction;

use crate::{crop, letterbox, resize};

use super::common::{
    crop_info_to_pydict, decode_image, encode_image, letterbox_info_to_pydict, map_crop_error,
    map_letterbox_error, map_resize_error, parse_filter, parse_format, rgb_image_from_numpy,
    rgb_image_to_numpy,
};

#[pyfunction(name = "compute_letterbox")]
fn compute_letterbox_py<'py>(
    py: Python<'py>,
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let info =
        letterbox::compute_letterbox(original_width, original_height, target_width, target_height)
            .map_err(map_letterbox_error)?;
    letterbox_info_to_pydict(py, info)
}

#[pyfunction]
#[pyo3(signature = (input_bytes, target_width, target_height, filter=None, output_format=None))]
fn resize_image<'py>(
    py: Python<'py>,
    input_bytes: &[u8],
    target_width: u32,
    target_height: u32,
    filter: Option<&str>,
    output_format: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let image = decode_image(input_bytes)?;
    let filter = parse_filter(filter)?;
    let output_format = parse_format(output_format)?;
    let result = resize::resize_image(&image, target_width, target_height, filter)
        .map_err(map_resize_error)?;
    let encoded = encode_image(DynamicImage::ImageRgb8(result.image), output_format)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (input_array, target_width, target_height, filter=None))]
fn resize_image_numpy<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray3<'py, u8>,
    target_width: u32,
    target_height: u32,
    filter: Option<&str>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = rgb_image_from_numpy(input_array)?;
    let filter = parse_filter(filter)?;
    let result = resize::resize_image(
        &DynamicImage::ImageRgb8(image),
        target_width,
        target_height,
        filter,
    )
    .map_err(map_resize_error)?;
    rgb_image_to_numpy(py, result.image)
}

#[pyfunction]
#[pyo3(signature = (input_bytes, x, y, width, height, output_format=None))]
fn crop_image<'py>(
    py: Python<'py>,
    input_bytes: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    output_format: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let image = decode_image(input_bytes)?;
    let output_format = parse_format(output_format)?;
    let result = crop::crop_image(&image, x, y, width, height).map_err(map_crop_error)?;
    let encoded = encode_image(DynamicImage::ImageRgb8(result.image), output_format)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (input_array, x, y, width, height))]
fn crop_image_numpy<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray3<'py, u8>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyDict>)> {
    let image = rgb_image_from_numpy(input_array)?;
    let result = crop::crop_image(&DynamicImage::ImageRgb8(image), x, y, width, height)
        .map_err(map_crop_error)?;
    let info = crop_info_to_pydict(py, result.info)?;
    let array = rgb_image_to_numpy(py, result.image)?;
    Ok((array, info))
}

#[pyfunction]
#[pyo3(signature = (input_bytes, width, height, output_format=None))]
fn center_crop_image<'py>(
    py: Python<'py>,
    input_bytes: &[u8],
    width: u32,
    height: u32,
    output_format: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let image = decode_image(input_bytes)?;
    let output_format = parse_format(output_format)?;
    let result = crop::center_crop_image(&image, width, height).map_err(map_crop_error)?;
    let encoded = encode_image(DynamicImage::ImageRgb8(result.image), output_format)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (input_array, width, height))]
fn center_crop_image_numpy<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray3<'py, u8>,
    width: u32,
    height: u32,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyDict>)> {
    let image = rgb_image_from_numpy(input_array)?;
    let result = crop::center_crop_image(&DynamicImage::ImageRgb8(image), width, height)
        .map_err(map_crop_error)?;
    let info = crop_info_to_pydict(py, result.info)?;
    let array = rgb_image_to_numpy(py, result.image)?;
    Ok((array, info))
}

#[pyfunction]
#[pyo3(signature = (
    input_bytes,
    target_width,
    target_height,
    fill=(114, 114, 114),
    filter=None,
    output_format=None
))]
fn letterbox_image<'py>(
    py: Python<'py>,
    input_bytes: &[u8],
    target_width: u32,
    target_height: u32,
    fill: (u8, u8, u8),
    filter: Option<&str>,
    output_format: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let image = decode_image(input_bytes)?;
    let filter = parse_filter(filter)?;
    let output_format = parse_format(output_format)?;
    let result = letterbox::letterbox_image(
        &image,
        target_width,
        target_height,
        [fill.0, fill.1, fill.2],
        filter,
    )
    .map_err(map_letterbox_error)?;
    let encoded = encode_image(DynamicImage::ImageRgb8(result.image), output_format)?;
    Ok(PyBytes::new(py, &encoded))
}

#[pyfunction]
#[pyo3(signature = (
    input_array,
    target_width,
    target_height,
    fill=(114, 114, 114),
    filter=None
))]
fn letterbox_image_numpy<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray3<'py, u8>,
    target_width: u32,
    target_height: u32,
    fill: (u8, u8, u8),
    filter: Option<&str>,
) -> PyResult<(Bound<'py, PyArray3<u8>>, Bound<'py, PyDict>)> {
    let image = rgb_image_from_numpy(input_array)?;
    let filter = parse_filter(filter)?;
    let result = letterbox::letterbox_image(
        &DynamicImage::ImageRgb8(image),
        target_width,
        target_height,
        [fill.0, fill.1, fill.2],
        filter,
    )
    .map_err(map_letterbox_error)?;
    let info = letterbox_info_to_pydict(py, result.info)?;
    let array = rgb_image_to_numpy(py, result.image)?;
    Ok((array, info))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_letterbox_py, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(center_crop_image, m)?)?;
    m.add_function(wrap_pyfunction!(center_crop_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_image, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_image_numpy, m)?)?;
    Ok(())
}
