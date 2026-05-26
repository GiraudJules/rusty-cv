use super::{
    batched_nms, batched_soft_nms, clip_boxes, filter_boxes_by_min_size,
    remap::remap_boxes_for_postprocess, BBoxError, BBoxXYXY, BoxRemap, Detection, NmsOptions,
    PostprocessOptions, PostprocessResult, SoftNmsOptions,
};

/// Run fused detection postprocessing over remapped boxes and class-aware NMS.
pub fn postprocess_detections(
    boxes: &[BBoxXYXY],
    scores: &[f32],
    class_ids: &[usize],
    remap: BoxRemap,
    options: &PostprocessOptions,
    soft_nms_options: Option<&SoftNmsOptions>,
) -> Result<PostprocessResult, BBoxError> {
    super::validate_postprocess_inputs(boxes, scores, class_ids, options)?;

    let (mut remapped_boxes, clip_bounds) = remap_boxes_for_postprocess(boxes, remap)?;
    if options.clip {
        if let Some((width, height)) = clip_bounds {
            remapped_boxes = clip_boxes(&remapped_boxes, width, height)?;
        }
    }

    let candidate_indices =
        filter_boxes_by_min_size(&remapped_boxes, options.min_width, options.min_height)?;
    let candidate_boxes = candidate_indices
        .iter()
        .map(|&index| remapped_boxes[index])
        .collect::<Vec<_>>();
    let candidate_scores = candidate_indices
        .iter()
        .map(|&index| scores[index])
        .collect::<Vec<_>>();
    let candidate_class_ids = candidate_indices
        .iter()
        .map(|&index| class_ids[index])
        .collect::<Vec<_>>();

    let detections = if let Some(soft_options) = soft_nms_options {
        batched_soft_nms(
            &candidate_boxes,
            &candidate_scores,
            &candidate_class_ids,
            soft_options,
        )?
    } else {
        let nms_options = NmsOptions {
            iou_threshold: options.iou_threshold,
            score_threshold: options.score_threshold,
            pre_nms_top_k: options.pre_nms_top_k,
            max_detections: options.max_detections,
        };
        batched_nms(
            &candidate_boxes,
            &candidate_scores,
            &candidate_class_ids,
            &nms_options,
        )?
    };

    let mut ordered_boxes = Vec::with_capacity(detections.len());
    let mut mapped_detections = Vec::with_capacity(detections.len());
    for detection in detections {
        let original_index = candidate_indices[detection.box_index];
        ordered_boxes.push(remapped_boxes[original_index]);
        mapped_detections.push(Detection {
            box_index: original_index,
            class_id: detection.class_id,
            score: detection.score,
        });
    }

    Ok(PostprocessResult {
        boxes: ordered_boxes,
        detections: mapped_detections,
    })
}
