use std::io::Cursor;

use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat, RgbImage};
use numpy::ndarray::{Array1, Array3};
use numpy::{PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

use crate::bbox::{
    self, BBoxError, BBoxXYXY, Detection, NmsOptions, SoftNmsMethod, SoftNmsOptions,
};
use crate::crop::{self, CropError};
use crate::letterbox::{self, LetterboxError};
use crate::normalize::{self, NormalizeError};
use crate::resize::{self, ResizeError};

fn parse_filter(filter: Option<&str>) -> PyResult<FilterType> {
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

fn parse_format(format: Option<&str>) -> PyResult<ImageFormat> {
    match format.unwrap_or("png").to_ascii_lowercase().as_str() {
        "png" => Ok(ImageFormat::Png),
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        other => Err(PyValueError::new_err(format!(
            "unsupported output format '{other}'. Use png or jpeg"
        ))),
    }
}

fn decode_image(input_bytes: &[u8]) -> PyResult<DynamicImage> {
    image::load_from_memory(input_bytes)
        .map_err(|err| PyValueError::new_err(format!("failed to decode image bytes: {err}")))
}

fn encode_image(image: DynamicImage, format: ImageFormat) -> PyResult<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, format)
        .map_err(|err| PyValueError::new_err(format!("failed to encode output image: {err}")))?;
    Ok(cursor.into_inner())
}

fn map_letterbox_error(err: LetterboxError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn map_bbox_error(err: BBoxError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn map_crop_error(err: CropError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn map_normalize_error(err: NormalizeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn map_resize_error(err: ResizeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn crop_info_to_pydict<'py>(
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

fn letterbox_info_to_pydict<'py>(
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

fn rgb_image_from_numpy(input: PyReadonlyArray3<'_, u8>) -> PyResult<RgbImage> {
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

fn float_array_to_numpy<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    height: u32,
    width: u32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let array = Array3::from_shape_vec((height as usize, width as usize, 3), data)
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray3::from_owned_array(py, array))
}

fn rgb_image_to_numpy<'py>(py: Python<'py>, image: RgbImage) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let (width, height) = image.dimensions();
    let array = Array3::from_shape_vec((height as usize, width as usize, 3), image.into_raw())
        .map_err(|err| PyValueError::new_err(format!("failed to build NumPy array: {err}")))?;
    Ok(PyArray3::from_owned_array(py, array))
}

fn boxes_from_numpy(input: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<BBoxXYXY>> {
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

fn scores_from_numpy(input: PyReadonlyArray1<'_, f32>) -> Vec<f32> {
    input.as_array().iter().copied().collect()
}

fn class_ids_from_numpy(input: PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
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

fn nms_options(
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

fn parse_soft_nms_method(method: Option<&str>) -> PyResult<SoftNmsMethod> {
    match method.unwrap_or("linear").to_ascii_lowercase().as_str() {
        "linear" => Ok(SoftNmsMethod::Linear),
        "gaussian" => Ok(SoftNmsMethod::Gaussian),
        other => Err(PyValueError::new_err(format!(
            "unsupported soft_nms method '{other}'. Use linear or gaussian"
        ))),
    }
}

fn soft_nms_options(
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

fn detections_to_pydict<'py>(
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

#[pyfunction(name = "iou")]
fn iou_py(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    bbox::iou(
        BBoxXYXY {
            x1: a.0,
            y1: a.1,
            x2: a.2,
            y2: a.3,
        },
        BBoxXYXY {
            x1: b.0,
            y1: b.1,
            x2: b.2,
            y2: b.3,
        },
    )
}

#[pyfunction(name = "nms")]
#[pyo3(signature = (
    boxes,
    scores,
    iou_threshold=0.5,
    score_threshold=None,
    pre_nms_top_k=None,
    max_detections=None
))]
fn nms_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> PyResult<Vec<usize>> {
    let boxes = boxes_from_numpy(boxes)?;
    let scores = scores_from_numpy(scores);
    let options = nms_options(
        iou_threshold,
        score_threshold,
        pre_nms_top_k,
        max_detections,
    );
    py.detach(move || bbox::nms_with_options(&boxes, &scores, &options))
        .map_err(map_bbox_error)
}

#[pyfunction(name = "batched_nms")]
#[pyo3(signature = (
    boxes,
    scores,
    class_ids,
    iou_threshold=0.5,
    score_threshold=None,
    pre_nms_top_k=None,
    max_detections=None
))]
fn batched_nms_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    class_ids: PyReadonlyArray1<'_, i64>,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let scores = scores_from_numpy(scores);
    let class_ids = class_ids_from_numpy(class_ids)?;
    let options = nms_options(
        iou_threshold,
        score_threshold,
        pre_nms_top_k,
        max_detections,
    );

    let detections = py
        .detach(move || bbox::batched_nms(&boxes, &scores, &class_ids, &options))
        .map_err(map_bbox_error)?;
    detections_to_pydict(py, detections)
}

#[pyfunction(name = "multiclass_nms")]
#[pyo3(signature = (
    boxes,
    class_scores,
    iou_threshold=0.5,
    score_threshold=None,
    pre_nms_top_k=None,
    max_detections=None
))]
fn multiclass_nms_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    class_scores: PyReadonlyArray2<'_, f32>,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let class_scores_array = class_scores.as_array();
    let (_, num_classes) = class_scores_array.dim();
    let class_scores = class_scores_array.iter().copied().collect::<Vec<f32>>();
    let options = nms_options(
        iou_threshold,
        score_threshold,
        pre_nms_top_k,
        max_detections,
    );

    let detections = py
        .detach(move || bbox::multiclass_nms(&boxes, &class_scores, num_classes, &options))
        .map_err(map_bbox_error)?;
    detections_to_pydict(py, detections)
}

#[pyfunction(name = "soft_nms")]
#[pyo3(signature = (
    boxes,
    scores,
    method=None,
    iou_threshold=0.5,
    score_threshold=None,
    sigma=0.5,
    pre_nms_top_k=None,
    max_detections=None
))]
fn soft_nms_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    method: Option<&str>,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    sigma: f32,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let scores = scores_from_numpy(scores);
    let method = parse_soft_nms_method(method)?;
    let options = soft_nms_options(
        method,
        iou_threshold,
        score_threshold,
        sigma,
        pre_nms_top_k,
        max_detections,
    );

    let detections = py
        .detach(move || bbox::soft_nms(&boxes, &scores, &options))
        .map_err(map_bbox_error)?;
    detections_to_pydict(py, detections)
}

#[pyfunction(name = "batched_soft_nms")]
#[pyo3(signature = (
    boxes,
    scores,
    class_ids,
    method=None,
    iou_threshold=0.5,
    score_threshold=None,
    sigma=0.5,
    pre_nms_top_k=None,
    max_detections=None
))]
fn batched_soft_nms_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    class_ids: PyReadonlyArray1<'_, i64>,
    method: Option<&str>,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    sigma: f32,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let scores = scores_from_numpy(scores);
    let class_ids = class_ids_from_numpy(class_ids)?;
    let method = parse_soft_nms_method(method)?;
    let options = soft_nms_options(
        method,
        iou_threshold,
        score_threshold,
        sigma,
        pre_nms_top_k,
        max_detections,
    );

    let detections = py
        .detach(move || bbox::batched_soft_nms(&boxes, &scores, &class_ids, &options))
        .map_err(map_bbox_error)?;
    detections_to_pydict(py, detections)
}

#[pyfunction(name = "multiclass_soft_nms")]
#[pyo3(signature = (
    boxes,
    class_scores,
    method=None,
    iou_threshold=0.5,
    score_threshold=None,
    sigma=0.5,
    pre_nms_top_k=None,
    max_detections=None
))]
fn multiclass_soft_nms_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    class_scores: PyReadonlyArray2<'_, f32>,
    method: Option<&str>,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    sigma: f32,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let class_scores_array = class_scores.as_array();
    let (_, num_classes) = class_scores_array.dim();
    let class_scores = class_scores_array.iter().copied().collect::<Vec<f32>>();
    let method = parse_soft_nms_method(method)?;
    let options = soft_nms_options(
        method,
        iou_threshold,
        score_threshold,
        sigma,
        pre_nms_top_k,
        max_detections,
    );

    let detections = py
        .detach(move || bbox::multiclass_soft_nms(&boxes, &class_scores, num_classes, &options))
        .map_err(map_bbox_error)?;
    detections_to_pydict(py, detections)
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
    float_array_to_numpy(py, result.data, result.info.height, result.info.width)
}

#[pymodule]
fn rusty_cv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_letterbox_py, m)?)?;
    m.add_function(wrap_pyfunction!(iou_py, m)?)?;
    m.add_function(wrap_pyfunction!(nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(batched_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(batched_soft_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_soft_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image, m)?)?;
    m.add_function(wrap_pyfunction!(crop_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(center_crop_image, m)?)?;
    m.add_function(wrap_pyfunction!(center_crop_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image, m)?)?;
    m.add_function(wrap_pyfunction!(resize_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_image, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_image_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_image_numpy, m)?)?;
    Ok(())
}
