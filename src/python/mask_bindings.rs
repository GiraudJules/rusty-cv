use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

use crate::mask;

use super::common::{
    grayscale_mask_f32_from_numpy, grayscale_mask_from_numpy, letterbox_info_to_pydict,
    map_mask_error, mask_u8_to_numpy,
};

#[pyfunction]
#[pyo3(signature = (mask_array, target_width, target_height))]
fn resize_mask_numpy<'py>(
    py: Python<'py>,
    mask_array: PyReadonlyArray2<'_, u8>,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let array = mask_array.as_array();
    let (height, width) = array.dim();
    let mask = grayscale_mask_from_numpy(mask_array);
    let result = mask::resize_mask(
        &mask,
        width as u32,
        height as u32,
        target_width,
        target_height,
    )
    .map_err(map_mask_error)?;
    mask_u8_to_numpy(py, result.data, result.height, result.width)
}

#[pyfunction]
#[pyo3(signature = (mask_array, target_width, target_height, fill=0))]
fn letterbox_mask_numpy<'py>(
    py: Python<'py>,
    mask_array: PyReadonlyArray2<'_, u8>,
    target_width: u32,
    target_height: u32,
    fill: u8,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyDict>)> {
    let array = mask_array.as_array();
    let (height, width) = array.dim();
    let mask = grayscale_mask_from_numpy(mask_array);
    let result = mask::letterbox_mask(
        &mask,
        width as u32,
        height as u32,
        target_width,
        target_height,
        fill,
    )
    .map_err(map_mask_error)?;
    let info = letterbox_info_to_pydict(py, result.info)?;
    let array = mask_u8_to_numpy(py, result.data, target_height, target_width)?;
    Ok((array, info))
}

#[pyfunction]
fn unletterbox_mask_numpy<'py>(
    py: Python<'py>,
    mask_array: PyReadonlyArray2<'_, u8>,
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let array = mask_array.as_array();
    let (height, width) = array.dim();
    if width as u32 != target_width || height as u32 != target_height {
        return Err(PyValueError::new_err(format!(
            "mask shape does not match target dimensions, got {}x{} and expected {}x{}",
            width, height, target_width, target_height
        )));
    }
    let mask = grayscale_mask_from_numpy(mask_array);
    let result = mask::unletterbox_mask(
        &mask,
        target_width,
        target_height,
        original_width,
        original_height,
    )
    .map_err(map_mask_error)?;
    mask_u8_to_numpy(py, result.data, result.height, result.width)
}

#[pyfunction]
#[pyo3(signature = (mask_array, threshold=0.5))]
fn threshold_mask_numpy<'py>(
    py: Python<'py>,
    mask_array: &Bound<'py, PyAny>,
    threshold: f32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    if let Ok(mask_array) = mask_array.extract::<PyReadonlyArray2<'_, f32>>() {
        let array = mask_array.as_array();
        let (height, width) = array.dim();
        let mask = grayscale_mask_f32_from_numpy(mask_array);
        let thresholded = mask::threshold_mask(&mask, width as u32, height as u32, threshold)
            .map_err(map_mask_error)?;
        return mask_u8_to_numpy(py, thresholded, height as u32, width as u32);
    }

    if let Ok(mask_array) = mask_array.extract::<PyReadonlyArray2<'_, u8>>() {
        let array = mask_array.as_array();
        let (height, width) = array.dim();
        let mask = mask_array
            .as_array()
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let thresholded = mask::threshold_mask(&mask, width as u32, height as u32, threshold)
            .map_err(map_mask_error)?;
        return mask_u8_to_numpy(py, thresholded, height as u32, width as u32);
    }

    Err(PyValueError::new_err(
        "expected a HxW NumPy array with dtype uint8 or float32",
    ))
}

#[pyfunction]
fn mask_to_box_numpy(
    mask_array: PyReadonlyArray2<'_, u8>,
) -> PyResult<Option<(f32, f32, f32, f32)>> {
    let array = mask_array.as_array();
    let (height, width) = array.dim();
    let mask = grayscale_mask_from_numpy(mask_array);
    let bbox = mask::mask_to_box(&mask, width as u32, height as u32).map_err(map_mask_error)?;
    Ok(bbox.map(|bbox| (bbox.x1, bbox.y1, bbox.x2, bbox.y2)))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resize_mask_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_mask_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(unletterbox_mask_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_mask_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(mask_to_box_numpy, m)?)?;
    Ok(())
}
