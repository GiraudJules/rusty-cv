use image::DynamicImage;
use numpy::{PyArray3, PyArray4, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;

use crate::{normalize, preprocess};

use super::common::{
    batch_float_array_to_numpy, float_array_to_numpy, map_normalize_error, map_preprocess_error,
    parse_filter, parse_preprocess_layout, parse_preprocess_mode, preprocess_info_to_pydict,
    preprocess_infos_to_pylist, rgb_image_from_numpy, rgb_images_from_python_input,
};

#[pyfunction]
#[pyo3(signature = (
    input_array,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    scale_to_unit=true
))]
fn normalize_image_numpy<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray3<'py, u8>,
    mean: (f32, f32, f32),
    std: (f32, f32, f32),
    scale_to_unit: bool,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let image = rgb_image_from_numpy(input_array)?;
    let result = normalize::normalize_image(
        &DynamicImage::ImageRgb8(image),
        [mean.0, mean.1, mean.2],
        [std.0, std.1, std.2],
        scale_to_unit,
    )
    .map_err(map_normalize_error)?;
    float_array_to_numpy(
        py,
        result.data,
        result.info.height,
        result.info.width,
        preprocess::PreprocessLayout::Hwc,
    )
}

#[pyfunction]
#[pyo3(signature = (
    input_array,
    target_width,
    target_height,
    mode=None,
    fill=(114, 114, 114),
    filter=None,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    scale_to_unit=true,
    layout=None
))]
#[allow(clippy::too_many_arguments)]
fn preprocess_image_numpy<'py>(
    py: Python<'py>,
    input_array: PyReadonlyArray3<'py, u8>,
    target_width: u32,
    target_height: u32,
    mode: Option<&str>,
    fill: (u8, u8, u8),
    filter: Option<&str>,
    mean: (f32, f32, f32),
    std: (f32, f32, f32),
    scale_to_unit: bool,
    layout: Option<&str>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyDict>)> {
    let image = rgb_image_from_numpy(input_array)?;
    let filter = parse_filter(filter)?;
    let mode = parse_preprocess_mode(mode, fill)?;
    let layout = parse_preprocess_layout(layout)?;
    let result = py
        .detach(move || {
            preprocess::preprocess_image(
                &DynamicImage::ImageRgb8(image),
                target_width,
                target_height,
                mode,
                filter,
                [mean.0, mean.1, mean.2],
                [std.0, std.1, std.2],
                scale_to_unit,
                layout,
            )
        })
        .map_err(map_preprocess_error)?;
    let info = preprocess_info_to_pydict(py, result.info)?;
    let array = float_array_to_numpy(
        py,
        result.data,
        result.info.height,
        result.info.width,
        layout,
    )?;
    Ok((array, info))
}

#[pyfunction]
#[pyo3(signature = (
    input_arrays,
    target_width,
    target_height,
    mode=None,
    fill=(114, 114, 114),
    filter=None,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    scale_to_unit=true,
    layout=None
))]
#[allow(clippy::too_many_arguments)]
fn preprocess_batch_numpy<'py>(
    py: Python<'py>,
    input_arrays: &Bound<'py, PyAny>,
    target_width: u32,
    target_height: u32,
    mode: Option<&str>,
    fill: (u8, u8, u8),
    filter: Option<&str>,
    mean: (f32, f32, f32),
    std: (f32, f32, f32),
    scale_to_unit: bool,
    layout: Option<&str>,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyList>)> {
    let images = rgb_images_from_python_input(input_arrays)?;
    let filter = parse_filter(filter)?;
    let mode = parse_preprocess_mode(mode, fill)?;
    let layout = parse_preprocess_layout(layout)?;
    let dynamic_images = images
        .into_iter()
        .map(DynamicImage::ImageRgb8)
        .collect::<Vec<_>>();

    let result = py
        .detach(move || {
            preprocess::preprocess_batch(
                &dynamic_images,
                target_width,
                target_height,
                mode,
                filter,
                [mean.0, mean.1, mean.2],
                [std.0, std.1, std.2],
                scale_to_unit,
                layout,
            )
        })
        .map_err(map_preprocess_error)?;

    let first_info = result.infos.first().copied().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "batch preprocessing requires at least one input image",
        )
    })?;
    let infos = preprocess_infos_to_pylist(py, result.infos)?;
    let array = batch_float_array_to_numpy(
        py,
        result.data,
        infos.len(),
        first_info.height,
        first_info.width,
        layout,
    )?;
    Ok((array, infos))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_batch_numpy, m)?)?;
    Ok(())
}
