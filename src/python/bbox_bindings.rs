use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

use crate::bbox::{self, BBoxXYXY, PostprocessOptions};

use super::common::{
    box_filter_result_to_pydict, boxes_from_numpy, boxes_xywh_from_numpy, boxes_xywh_to_numpy,
    boxes_xyxy_to_numpy, class_ids_from_numpy, detections_to_pydict, filtered_indices_to_pydict,
    map_bbox_error, nms_options, parse_postprocess_remap, parse_soft_nms_method,
    postprocess_result_to_pydict, scores_from_numpy, soft_nms_options,
};

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

#[pyfunction]
fn xyxy_to_xywh_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes_from_numpy(boxes)?;
    let converted = bbox::xyxy_to_xywh(&boxes).map_err(map_bbox_error)?;
    boxes_xywh_to_numpy(py, converted)
}

#[pyfunction]
fn xywh_to_xyxy_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes_xywh_from_numpy(boxes)?;
    let converted = bbox::xywh_to_xyxy(&boxes).map_err(map_bbox_error)?;
    boxes_xyxy_to_numpy(py, converted)
}

#[pyfunction]
fn clip_boxes_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes_from_numpy(boxes)?;
    let clipped = bbox::clip_boxes(&boxes, width, height).map_err(map_bbox_error)?;
    boxes_xyxy_to_numpy(py, clipped)
}

#[pyfunction]
fn filter_boxes_by_score_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    threshold: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let scores = scores_from_numpy(scores);
    let kept = bbox::filter_boxes_by_score(&boxes, &scores, threshold).map_err(map_bbox_error)?;
    filtered_indices_to_pydict(py, &boxes, kept, Some(&scores))
}

#[pyfunction]
#[pyo3(signature = (boxes, min_area=None, max_area=None))]
fn filter_boxes_by_area_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    min_area: Option<f32>,
    max_area: Option<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let kept = bbox::filter_boxes_by_area(&boxes, min_area, max_area).map_err(map_bbox_error)?;
    filtered_indices_to_pydict(py, &boxes, kept, None)
}

#[pyfunction]
fn filter_boxes_by_min_size_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    min_width: f32,
    min_height: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let kept =
        bbox::filter_boxes_by_min_size(&boxes, min_width, min_height).map_err(map_bbox_error)?;
    filtered_indices_to_pydict(py, &boxes, kept, None)
}

#[pyfunction]
#[pyo3(signature = (boxes, width, height, min_width=0.0, min_height=0.0))]
fn clip_and_filter_boxes_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    min_width: f32,
    min_height: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let result_data = bbox::clip_and_filter_boxes(&boxes, width, height, min_width, min_height)
        .map_err(map_bbox_error)?;
    box_filter_result_to_pydict(py, result_data)
}

#[pyfunction]
fn resize_boxes_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes_from_numpy(boxes)?;
    let resized = bbox::resize_boxes(
        &boxes,
        original_width,
        original_height,
        target_width,
        target_height,
    )
    .map_err(map_bbox_error)?;
    boxes_xyxy_to_numpy(py, resized)
}

#[pyfunction]
fn letterbox_boxes_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes_from_numpy(boxes)?;
    let remapped = bbox::letterbox_boxes(
        &boxes,
        original_width,
        original_height,
        target_width,
        target_height,
    )
    .map_err(map_bbox_error)?;
    boxes_xyxy_to_numpy(py, remapped)
}

#[pyfunction]
fn unletterbox_boxes_numpy<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    original_width: u32,
    original_height: u32,
    target_width: u32,
    target_height: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let boxes = boxes_from_numpy(boxes)?;
    let remapped = bbox::unletterbox_boxes(
        &boxes,
        original_width,
        original_height,
        target_width,
        target_height,
    )
    .map_err(map_bbox_error)?;
    boxes_xyxy_to_numpy(py, remapped)
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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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

#[pyfunction(name = "postprocess_detections")]
#[pyo3(signature = (
    boxes,
    scores,
    class_ids=None,
    geometry_mode=None,
    processed_width=None,
    processed_height=None,
    original_width=None,
    original_height=None,
    clip=false,
    iou_threshold=0.5,
    score_threshold=None,
    pre_nms_top_k=None,
    max_detections=None,
    min_width=0.0,
    min_height=0.0,
    soft=false,
    soft_method=None,
    sigma=0.5
))]
#[allow(clippy::too_many_arguments)]
fn postprocess_detections_py<'py>(
    py: Python<'py>,
    boxes: PyReadonlyArray2<'_, f32>,
    scores: PyReadonlyArray1<'_, f32>,
    class_ids: Option<PyReadonlyArray1<'_, i64>>,
    geometry_mode: Option<&str>,
    processed_width: Option<u32>,
    processed_height: Option<u32>,
    original_width: Option<u32>,
    original_height: Option<u32>,
    clip: bool,
    iou_threshold: f32,
    score_threshold: Option<f32>,
    pre_nms_top_k: Option<usize>,
    max_detections: Option<usize>,
    min_width: f32,
    min_height: f32,
    soft: bool,
    soft_method: Option<&str>,
    sigma: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let boxes = boxes_from_numpy(boxes)?;
    let scores = scores_from_numpy(scores);
    let class_ids = if let Some(class_ids) = class_ids {
        class_ids_from_numpy(class_ids)?
    } else {
        vec![0usize; boxes.len()]
    };
    let remap = parse_postprocess_remap(
        geometry_mode,
        processed_width,
        processed_height,
        original_width,
        original_height,
    )?;
    let options = PostprocessOptions {
        iou_threshold,
        score_threshold: score_threshold.unwrap_or(f32::NEG_INFINITY),
        pre_nms_top_k,
        max_detections,
        min_width,
        min_height,
        clip,
    };
    let soft_options = if soft {
        Some(soft_nms_options(
            parse_soft_nms_method(soft_method)?,
            iou_threshold,
            score_threshold,
            sigma,
            pre_nms_top_k,
            max_detections,
        ))
    } else {
        None
    };

    let result_data = py
        .detach(move || {
            bbox::postprocess_detections(
                &boxes,
                &scores,
                &class_ids,
                remap,
                &options,
                soft_options.as_ref(),
            )
        })
        .map_err(map_bbox_error)?;
    postprocess_result_to_pydict(py, result_data)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(iou_py, m)?)?;
    m.add_function(wrap_pyfunction!(xyxy_to_xywh_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(xywh_to_xyxy_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(clip_boxes_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(filter_boxes_by_score_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(filter_boxes_by_area_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(filter_boxes_by_min_size_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(clip_and_filter_boxes_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(resize_boxes_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(letterbox_boxes_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(unletterbox_boxes_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(batched_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(batched_soft_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(multiclass_soft_nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(postprocess_detections_py, m)?)?;
    Ok(())
}
