use numpy::{PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

use crate::layout;

use super::common::{array3_to_pyobject, array4_to_pyobject, map_layout_error};

#[pyfunction]
fn hwc_to_chw_numpy<'py>(py: Python<'py>, input_array: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(array) = input_array.extract::<PyReadonlyArray3<'_, u8>>() {
        let array_view = array.as_array();
        let (height, width, channels) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::hwc_to_chw(&data, height as u32, width as u32, channels as u32)
            .map_err(map_layout_error)?;
        return array3_to_pyobject(py, converted, (channels, height, width));
    }

    if let Ok(array) = input_array.extract::<PyReadonlyArray3<'_, f32>>() {
        let array_view = array.as_array();
        let (height, width, channels) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::hwc_to_chw(&data, height as u32, width as u32, channels as u32)
            .map_err(map_layout_error)?;
        return array3_to_pyobject(py, converted, (channels, height, width));
    }

    Err(PyValueError::new_err(
        "expected a HxWxC NumPy array with dtype uint8 or float32",
    ))
}

#[pyfunction]
fn chw_to_hwc_numpy<'py>(py: Python<'py>, input_array: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(array) = input_array.extract::<PyReadonlyArray3<'_, u8>>() {
        let array_view = array.as_array();
        let (channels, height, width) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::chw_to_hwc(&data, channels as u32, height as u32, width as u32)
            .map_err(map_layout_error)?;
        return array3_to_pyobject(py, converted, (height, width, channels));
    }

    if let Ok(array) = input_array.extract::<PyReadonlyArray3<'_, f32>>() {
        let array_view = array.as_array();
        let (channels, height, width) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::chw_to_hwc(&data, channels as u32, height as u32, width as u32)
            .map_err(map_layout_error)?;
        return array3_to_pyobject(py, converted, (height, width, channels));
    }

    Err(PyValueError::new_err(
        "expected a CxHxW NumPy array with dtype uint8 or float32",
    ))
}

#[pyfunction]
fn rgb_to_bgr_numpy<'py>(py: Python<'py>, input_array: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(array) = input_array.extract::<PyReadonlyArray3<'_, u8>>() {
        let array_view = array.as_array();
        let (height, width, channels) = array_view.dim();
        if channels != 3 {
            return Err(PyValueError::new_err(format!(
                "expected a HxWx3 NumPy array, got last dimension {}",
                channels
            )));
        }
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted =
            layout::rgb_to_bgr(&data, height as u32, width as u32).map_err(map_layout_error)?;
        return array3_to_pyobject(py, converted, (height, width, channels));
    }

    if let Ok(array) = input_array.extract::<PyReadonlyArray3<'_, f32>>() {
        let array_view = array.as_array();
        let (height, width, channels) = array_view.dim();
        if channels != 3 {
            return Err(PyValueError::new_err(format!(
                "expected a HxWx3 NumPy array, got last dimension {}",
                channels
            )));
        }
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted =
            layout::rgb_to_bgr(&data, height as u32, width as u32).map_err(map_layout_error)?;
        return array3_to_pyobject(py, converted, (height, width, channels));
    }

    Err(PyValueError::new_err(
        "expected a HxWx3 NumPy array with dtype uint8 or float32",
    ))
}

#[pyfunction]
fn nhwc_to_nchw_numpy<'py>(
    py: Python<'py>,
    input_array: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    if let Ok(array) = input_array.extract::<PyReadonlyArray4<'_, u8>>() {
        let array_view = array.as_array();
        let (batch, height, width, channels) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::nhwc_to_nchw(
            &data,
            batch as u32,
            height as u32,
            width as u32,
            channels as u32,
        )
        .map_err(map_layout_error)?;
        return array4_to_pyobject(py, converted, (batch, channels, height, width));
    }

    if let Ok(array) = input_array.extract::<PyReadonlyArray4<'_, f32>>() {
        let array_view = array.as_array();
        let (batch, height, width, channels) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::nhwc_to_nchw(
            &data,
            batch as u32,
            height as u32,
            width as u32,
            channels as u32,
        )
        .map_err(map_layout_error)?;
        return array4_to_pyobject(py, converted, (batch, channels, height, width));
    }

    Err(PyValueError::new_err(
        "expected a NxHxWxC NumPy array with dtype uint8 or float32",
    ))
}

#[pyfunction]
fn nchw_to_nhwc_numpy<'py>(
    py: Python<'py>,
    input_array: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    if let Ok(array) = input_array.extract::<PyReadonlyArray4<'_, u8>>() {
        let array_view = array.as_array();
        let (batch, channels, height, width) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::nchw_to_nhwc(
            &data,
            batch as u32,
            channels as u32,
            height as u32,
            width as u32,
        )
        .map_err(map_layout_error)?;
        return array4_to_pyobject(py, converted, (batch, height, width, channels));
    }

    if let Ok(array) = input_array.extract::<PyReadonlyArray4<'_, f32>>() {
        let array_view = array.as_array();
        let (batch, channels, height, width) = array_view.dim();
        let data = array_view.iter().copied().collect::<Vec<_>>();
        let converted = layout::nchw_to_nhwc(
            &data,
            batch as u32,
            channels as u32,
            height as u32,
            width as u32,
        )
        .map_err(map_layout_error)?;
        return array4_to_pyobject(py, converted, (batch, height, width, channels));
    }

    Err(PyValueError::new_err(
        "expected a NxCxHxW NumPy array with dtype uint8 or float32",
    ))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hwc_to_chw_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(chw_to_hwc_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_bgr_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(nhwc_to_nchw_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(nchw_to_nhwc_numpy, m)?)?;
    Ok(())
}
