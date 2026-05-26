use std::io::Cursor;

use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat, RgbImage};
use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArray4,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::bbox::{
    self, BBoxError, BBoxXYWH, BBoxXYXY, BoxRemap, Detection, NmsOptions, SoftNmsMethod,
    SoftNmsOptions,
};
use crate::crop::CropError;
use crate::layout::LayoutError;
use crate::letterbox::LetterboxError;
use crate::mask::MaskError;
use crate::normalize::NormalizeError;
use crate::preprocess::{
    self, PreprocessError, PreprocessGeometry, PreprocessLayout, PreprocessMode,
};
use crate::resize::ResizeError;

pub(super) fn parse_filter(filter: Option<&str>) -> PyResult<FilterType> {
    match filter.unwrap_or("triangle").to_ascii_lowercase().as_str() {
        "nearest" => Ok(FilterType::Nearest),
        "triangle" | "bilinear" => Ok(FilterType::Triangle),
        "catmull_rom" | "catmull-rom" => Ok(FilterType::CatmullRom),
        "gaussian" => Ok(FilterType::Gaussian),
        "lanczos3" | "lanczos" => Ok(FilterType::Lanczos3),
        other => Err(PyValueError::new_err(format!(
            "unsupported filter '{other}'. Use nearest, triangle, catmull_rom, gaussian, or lanczos3"
        ))),
    }
}

pub(super) fn parse_format(format: Option<&str>) -> PyResult<ImageFormat> {
    match format.unwrap_or("png").to_ascii_lowercase().as_str() {
        "png" => Ok(ImageFormat::Png),
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        other => Err(PyValueError::new_err(format!(
            "unsupported output format '{other}'. Use png or jpeg"
        ))),
    }
}

pub(super) fn decode_image(input_bytes: &[u8]) -> PyResult<DynamicImage> {
    image::load_from_memory(input_bytes)
        .map_err(|err| PyValueError::new_err(format!("failed to decode image bytes: {err}")))
}

pub(super) fn encode_image(image: DynamicImage, format: ImageFormat) -> PyResult<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, format)
        .map_err(|err| PyValueError::new_err(format!("failed to encode output image: {err}")))?;
    Ok(cursor.into_inner())
}

pub(super) fn map_letterbox_error(err: LetterboxError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_bbox_error(err: BBoxError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_crop_error(err: CropError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_layout_error(err: LayoutError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_mask_error(err: MaskError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_normalize_error(err: NormalizeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_preprocess_error(err: PreprocessError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn map_resize_error(err: ResizeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(super) fn crop_info_to_pydict<'py>(
    py: Python<'py>,
    info: crate::crop::CropInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("original_width", info.original_width)?;
    result.set_item("original_height", info.original_height)?;
    result.set_item("x", info.x)?;
    result.set_item("y", info.y)?;
    result.set_item("width", info.width)?;
    result.set_item("height", info.height)?;
    Ok(result)
}

pub(super) fn letterbox_info_to_pydict<'py>(
    py: Python<'py>,
    info: crate::letterbox::LetterboxInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let padding = PyDict::new(py);
    padding.set_item("top", info.padding.top)?;
    padding.set_item("bottom", info.padding.bottom)?;
    padding.set_item("left", info.padding.left)?;
    padding.set_item("right", info.padding.right)?;

    let result = PyDict::new(py);
    result.set_item("original_width", info.original_width)?;
    result.set_item("original_height", info.original_height)?;
    result.set_item("target_width", info.target_width)?;
    result.set_item("target_height", info.target_height)?;
    result.set_item("resized_width", info.resized_width)?;
    result.set_item("resized_height", info.resized_height)?;
    result.set_item("scale", info.scale)?;
    result.set_item("padding", padding)?;
    Ok(result)
}

pub(super) fn preprocess_info_to_pydict<'py>(
    py: Python<'py>,
    info: preprocess::PreprocessInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("width", info.width)?;
    result.set_item("height", info.height)?;
    result.set_item("channels", info.channels)?;
    result.set_item("scale_to_unit", info.scale_to_unit)?;
    result.set_item(
        "layout",
        match info.layout {
            PreprocessLayout::Hwc => "hwc",
            PreprocessLayout::Chw => "chw",
        },
    )?;

    match info.geometry {
        PreprocessGeometry::Resize(resize_info) => {
            let geometry = PyDict::new(py);
            geometry.set_item("original_width", resize_info.original_width)?;
            geometry.set_item("original_height", resize_info.original_height)?;
            geometry.set_item("target_width", resize_info.target_width)?;
            geometry.set_item("target_height", resize_info.target_height)?;
            result.set_item("mode", "resize")?;
            result.set_item("geometry", geometry)?;
        }
        PreprocessGeometry::Letterbox(letterbox_info) => {
            result.set_item("mode", "letterbox")?;
            result.set_item("geometry", letterbox_info_to_pydict(py, letterbox_info)?)?;
        }
    }

    Ok(result)
}

pub(super) fn preprocess_infos_to_pylist<'py>(
    py: Python<'py>,
    infos: Vec<preprocess::PreprocessInfo>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for info in infos {
        list.append(preprocess_info_to_pydict(py, info)?)?;
    }
    Ok(list)
}

pub(super) fn grayscale_mask_from_numpy(input: PyReadonlyArray2<'_, u8>) -> Vec<u8> {
    input.as_array().iter().copied().collect()
}

pub(super) fn grayscale_mask_f32_from_numpy(input: PyReadonlyArray2<'_, f32>) -> Vec<f32> {
    input.as_array().iter().copied().collect()
}

pub(super) fn mask_u8_to_numpy<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    height: u32,
    width: u32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let array = Array2::from_shape_vec((height as usize, width as usize), data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray2::from_owned_array(py, array))
}

pub(super) fn rgb_image_from_numpy(input: PyReadonlyArray3<'_, u8>) -> PyResult<RgbImage> {
    let array = input.as_array();
    let (height, width, channels) = array.dim();

    if channels != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a HxWx3 uint8 array, got last dimension {channels}"
        )));
    }

    let mut buffer = Vec::with_capacity(height * width * channels);
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                buffer.push(array[(y, x, c)]);
            }
        }
    }

    RgbImage::from_vec(width as u32, height as u32, buffer).ok_or_else(|| {
        PyValueError::new_err("failed to convert NumPy array into an RGB image buffer")
    })
}

pub(super) fn rgb_images_from_python_input(input: &Bound<'_, PyAny>) -> PyResult<Vec<RgbImage>> {
    if let Ok(array) = input.extract::<PyReadonlyArray4<'_, u8>>() {
        let array = array.as_array();
        let (batch, height, width, channels) = array.dim();

        if channels != 3 {
            return Err(PyValueError::new_err(format!(
                "expected a NxHxWx3 uint8 array, got last dimension {channels}"
            )));
        }

        let mut images = Vec::with_capacity(batch);
        for index in 0..batch {
            let mut buffer = Vec::with_capacity(height * width * channels);
            for y in 0..height {
                for x in 0..width {
                    for c in 0..channels {
                        buffer.push(array[(index, y, x, c)]);
                    }
                }
            }

            let image =
                RgbImage::from_vec(width as u32, height as u32, buffer).ok_or_else(|| {
                    PyValueError::new_err("failed to convert batched NumPy array into RGB images")
                })?;
            images.push(image);
        }

        return Ok(images);
    }

    let iterator = input.try_iter().map_err(|_| {
        PyValueError::new_err(
            "expected either a NxHxWx3 uint8 array or a sequence of HxWx3 uint8 arrays",
        )
    })?;

    let mut images = Vec::new();
    for item in iterator {
        let array = item?.extract::<PyReadonlyArray3<'_, u8>>().map_err(|_| {
            PyValueError::new_err(
                "all sequence items must be HxWx3 uint8 arrays for batch preprocessing",
            )
        })?;
        images.push(rgb_image_from_numpy(array)?);
    }

    if images.is_empty() {
        return Err(PyValueError::new_err(
            "batch preprocessing requires at least one input image",
        ));
    }

    Ok(images)
}

pub(super) fn float_array_to_numpy<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    height: u32,
    width: u32,
    layout: PreprocessLayout,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = match layout {
        PreprocessLayout::Hwc => (height as usize, width as usize, 3),
        PreprocessLayout::Chw => (3, height as usize, width as usize),
    };
    let array = Array3::from_shape_vec(shape, data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray3::from_owned_array(py, array))
}

pub(super) fn batch_float_array_to_numpy<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    batch: usize,
    height: u32,
    width: u32,
    layout: PreprocessLayout,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = match layout {
        PreprocessLayout::Hwc => (batch, height as usize, width as usize, 3),
        PreprocessLayout::Chw => (batch, 3, height as usize, width as usize),
    };
    let array = Array4::from_shape_vec(shape, data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray4::from_owned_array(py, array))
}

pub(super) fn rgb_image_to_numpy<'py>(
    py: Python<'py>,
    image: RgbImage,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let (width, height) = image.dimensions();
    let array = Array3::from_shape_vec((height as usize, width as usize, 3), image.into_raw())
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray3::from_owned_array(py, array))
}

pub(super) fn boxes_from_numpy(input: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<BBoxXYXY>> {
    let array = input.as_array();
    let (rows, cols) = array.dim();

    if cols != 4 {
        return Err(PyValueError::new_err(format!(
            "expected a Nx4 float32 array for boxes, got shape (_, {cols})"
        )));
    }

    let mut boxes = Vec::with_capacity(rows);
    for row in 0..rows {
        boxes.push(BBoxXYXY {
            x1: array[(row, 0)],
            y1: array[(row, 1)],
            x2: array[(row, 2)],
            y2: array[(row, 3)],
        });
    }
    Ok(boxes)
}

pub(super) fn boxes_xywh_from_numpy(input: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<BBoxXYWH>> {
    let array = input.as_array();
    let (rows, cols) = array.dim();

    if cols != 4 {
        return Err(PyValueError::new_err(format!(
            "expected a Nx4 float32 array for boxes, got shape (_, {cols})"
        )));
    }

    let mut boxes = Vec::with_capacity(rows);
    for row in 0..rows {
        boxes.push(BBoxXYWH {
            x: array[(row, 0)],
            y: array[(row, 1)],
            width: array[(row, 2)],
            height: array[(row, 3)],
        });
    }
    Ok(boxes)
}

pub(super) fn boxes_xyxy_to_numpy<'py>(
    py: Python<'py>,
    boxes: Vec<BBoxXYXY>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut data = Vec::with_capacity(boxes.len() * 4);
    for bbox in boxes {
        data.extend_from_slice(&[bbox.x1, bbox.y1, bbox.x2, bbox.y2]);
    }
    let array = Array2::from_shape_vec((data.len() / 4, 4), data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray2::from_owned_array(py, array))
}

pub(super) fn boxes_xywh_to_numpy<'py>(
    py: Python<'py>,
    boxes: Vec<BBoxXYWH>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut data = Vec::with_capacity(boxes.len() * 4);
    for bbox in boxes {
        data.extend_from_slice(&[bbox.x, bbox.y, bbox.width, bbox.height]);
    }
    let array = Array2::from_shape_vec((data.len() / 4, 4), data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray2::from_owned_array(py, array))
}

pub(super) fn array3_to_pyobject<T: numpy::Element>(
    py: Python<'_>,
    data: Vec<T>,
    shape: (usize, usize, usize),
) -> PyResult<Py<PyAny>> {
    let array = Array3::from_shape_vec(shape, data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray3::from_owned_array(py, array).into_any().unbind())
}

pub(super) fn array4_to_pyobject<T: numpy::Element>(
    py: Python<'_>,
    data: Vec<T>,
    shape: (usize, usize, usize, usize),
) -> PyResult<Py<PyAny>> {
    let array = Array4::from_shape_vec(shape, data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray4::from_owned_array(py, array).into_any().unbind())
}

pub(super) fn scores_from_numpy(input: PyReadonlyArray1<'_, f32>) -> Vec<f32> {
    input.as_array().iter().copied().collect()
}

pub(super) fn class_ids_from_numpy(input: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    let mut class_ids = Vec::with_capacity(input.len()?);
    for value in input.as_array().iter().copied() {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "class_ids must be non-negative, got {}",
                value
            )));
        }
        class_ids.push(value as usize);
    }
    Ok(class_ids)
}

pub(super) fn nms_options(
    iou_threshold: f32,
    score_threshold: Option<f32>,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> NmsOptions {
    NmsOptions {
        iou_threshold,
        score_threshold: score_threshold.unwrap_or(f32::NEG_INFINITY),
        pre_nms_top_k,
        max_detections,
    }
}

pub(super) fn parse_soft_nms_method(method: Option<&str>) -> PyResult<SoftNmsMethod> {
    match method.unwrap_or("linear").to_ascii_lowercase().as_str() {
        "linear" => Ok(SoftNmsMethod::Linear),
        "gaussian" => Ok(SoftNmsMethod::Gaussian),
        other => Err(PyValueError::new_err(format!(
            "unsupported soft_nms method '{other}'. Use linear or gaussian"
        ))),
    }
}

pub(super) fn parse_preprocess_layout(layout: Option<&str>) -> PyResult<PreprocessLayout> {
    match layout.unwrap_or("chw").to_ascii_lowercase().as_str() {
        "chw" => Ok(PreprocessLayout::Chw),
        "hwc" => Ok(PreprocessLayout::Hwc),
        other => Err(PyValueError::new_err(format!(
            "unsupported layout '{other}'. Use chw or hwc"
        ))),
    }
}

pub(super) fn parse_preprocess_mode(
    mode: Option<&str>,
    fill: (u8, u8, u8),
) -> PyResult<PreprocessMode> {
    match mode.unwrap_or("letterbox").to_ascii_lowercase().as_str() {
        "resize" => Ok(PreprocessMode::Resize),
        "letterbox" => Ok(PreprocessMode::Letterbox {
            fill: [fill.0, fill.1, fill.2],
        }),
        other => Err(PyValueError::new_err(format!(
            "unsupported mode '{other}'. Use resize or letterbox"
        ))),
    }
}

pub(super) fn soft_nms_options(
    method: SoftNmsMethod,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    sigma: f32,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> SoftNmsOptions {
    SoftNmsOptions {
        method,
        iou_threshold,
        score_threshold: score_threshold.unwrap_or(f32::NEG_INFINITY),
        sigma,
        pre_nms_top_k,
        max_detections,
    }
}

pub(super) fn detections_to_pydict<'py>(
    py: Python<'py>,
    detections: Vec<Detection>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut indices = Vec::with_capacity(detections.len());
    let mut class_ids = Vec::with_capacity(detections.len());
    let mut scores = Vec::with_capacity(detections.len());

    for detection in detections {
        indices.push(detection.box_index as i64);
        class_ids.push(detection.class_id as i64);
        scores.push(detection.score);
    }

    let result = PyDict::new(py);
    result.set_item(
        "indices",
        PyArray1::from_owned_array(py, Array1::from_vec(indices)),
    )?;
    result.set_item(
        "class_ids",
        PyArray1::from_owned_array(py, Array1::from_vec(class_ids)),
    )?;
    result.set_item(
        "scores",
        PyArray1::from_owned_array(py, Array1::from_vec(scores)),
    )?;
    Ok(result)
}

pub(super) fn filtered_indices_to_pydict<'py>(
    py: Python<'py>,
    boxes: &[BBoxXYXY],
    indices: Vec<usize>,
    scores: Option<&[f32]>,
) -> PyResult<Bound<'py, PyDict>> {
    let selected_boxes = indices
        .iter()
        .map(|&index| boxes[index])
        .collect::<Vec<_>>();
    let selected_indices = indices
        .iter()
        .map(|&index| index as i64)
        .collect::<Vec<_>>();

    let result = PyDict::new(py);
    result.set_item(
        "indices",
        PyArray1::from_owned_array(py, Array1::from_vec(selected_indices)),
    )?;
    result.set_item("boxes", boxes_xyxy_to_numpy(py, selected_boxes)?)?;

    if let Some(scores) = scores {
        let selected_scores = indices
            .iter()
            .map(|&index| scores[index])
            .collect::<Vec<_>>();
        result.set_item(
            "scores",
            PyArray1::from_owned_array(py, Array1::from_vec(selected_scores)),
        )?;
    }

    Ok(result)
}

pub(super) fn box_filter_result_to_pydict<'py>(
    py: Python<'py>,
    result_data: bbox::BoxFilterResult,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let indices = result_data
        .indices
        .iter()
        .map(|&index| index as i64)
        .collect::<Vec<_>>();
    result.set_item(
        "indices",
        PyArray1::from_owned_array(py, Array1::from_vec(indices)),
    )?;
    result.set_item("boxes", boxes_xyxy_to_numpy(py, result_data.boxes)?)?;
    Ok(result)
}

pub(super) fn postprocess_result_to_pydict<'py>(
    py: Python<'py>,
    result_data: bbox::PostprocessResult,
) -> PyResult<Bound<'py, PyDict>> {
    let mut indices = Vec::with_capacity(result_data.detections.len());
    let mut class_ids = Vec::with_capacity(result_data.detections.len());
    let mut scores = Vec::with_capacity(result_data.detections.len());

    for detection in result_data.detections {
        indices.push(detection.box_index as i64);
        class_ids.push(detection.class_id as i64);
        scores.push(detection.score);
    }

    let result = PyDict::new(py);
    result.set_item("boxes", boxes_xyxy_to_numpy(py, result_data.boxes)?)?;
    result.set_item(
        "indices",
        PyArray1::from_owned_array(py, Array1::from_vec(indices)),
    )?;
    result.set_item(
        "class_ids",
        PyArray1::from_owned_array(py, Array1::from_vec(class_ids)),
    )?;
    result.set_item(
        "scores",
        PyArray1::from_owned_array(py, Array1::from_vec(scores)),
    )?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_postprocess_remap(
    geometry_mode: Option<&str>,
    processed_width: Option<u32>,
    processed_height: Option<u32>,
    original_width: Option<u32>,
    original_height: Option<u32>,
) -> PyResult<BoxRemap> {
    match geometry_mode.map(|value| value.to_ascii_lowercase()) {
        None => Ok(BoxRemap::None),
        Some(mode) if mode == "current" => Ok(BoxRemap::Current {
            width: processed_width.ok_or_else(|| {
                PyValueError::new_err("processed_width is required for geometry_mode='current'")
            })?,
            height: processed_height.ok_or_else(|| {
                PyValueError::new_err("processed_height is required for geometry_mode='current'")
            })?,
        }),
        Some(mode) if mode == "resize" => Ok(BoxRemap::Resize {
            processed_width: processed_width.ok_or_else(|| {
                PyValueError::new_err("processed_width is required for geometry_mode='resize'")
            })?,
            processed_height: processed_height.ok_or_else(|| {
                PyValueError::new_err("processed_height is required for geometry_mode='resize'")
            })?,
            original_width: original_width.ok_or_else(|| {
                PyValueError::new_err("original_width is required for geometry_mode='resize'")
            })?,
            original_height: original_height.ok_or_else(|| {
                PyValueError::new_err("original_height is required for geometry_mode='resize'")
            })?,
        }),
        Some(mode) if mode == "letterbox" => Ok(BoxRemap::Letterbox {
            processed_width: processed_width.ok_or_else(|| {
                PyValueError::new_err("processed_width is required for geometry_mode='letterbox'")
            })?,
            processed_height: processed_height.ok_or_else(|| {
                PyValueError::new_err("processed_height is required for geometry_mode='letterbox'")
            })?,
            original_width: original_width.ok_or_else(|| {
                PyValueError::new_err("original_width is required for geometry_mode='letterbox'")
            })?,
            original_height: original_height.ok_or_else(|| {
                PyValueError::new_err("original_height is required for geometry_mode='letterbox'")
            })?,
        }),
        Some(mode) => Err(PyValueError::new_err(format!(
            "unsupported geometry_mode '{mode}'. Use current, resize, or letterbox"
        ))),
    }
}
