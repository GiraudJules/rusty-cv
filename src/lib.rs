//! `rusty-cv` is a small Rust-first computer vision crate focused on simple,
//! reusable image preprocessing primitives.
//!
//! Current functionality:
//!
//! - box geometry helpers, IoU, hard NMS variants, and soft NMS variants
//! - direct crop and center crop
//! - direct resize to exact dimensions
//! - letterbox resize with aspect-ratio preservation and padding
//! - RGB normalization and fused preprocessing into contiguous `f32` buffers
//! - optional Python bindings behind the `python` feature

/// Bounding-box geometry and postprocessing operations.
pub mod bbox;
/// Crop geometry and image operations.
pub mod crop;
/// Tensor layout and channel-order helpers.
pub mod layout;
/// Letterbox geometry and image operations.
pub mod letterbox;
/// Segmentation mask geometry helpers.
pub mod mask;
/// Image normalization operations.
pub mod normalize;
/// Fused inference preprocessing operations.
pub mod preprocess;
/// Direct image resize operations.
pub mod resize;

#[cfg(feature = "python")]
mod python;

/// Error returned by box postprocessing operations.
pub use bbox::{
    batched_nms, batched_soft_nms, clip_and_filter_boxes, clip_boxes, filter_boxes_by_area,
    filter_boxes_by_min_size, filter_boxes_by_score, iou, letterbox_boxes, multiclass_nms,
    multiclass_soft_nms, nms, nms_with_options, postprocess_detections, resize_boxes, soft_nms,
    unletterbox_boxes, xywh_to_xyxy, xyxy_to_xywh, BBoxError, BBoxXYWH, BBoxXYXY, BoxFilterResult,
    BoxRemap, Detection, NmsOptions, PostprocessOptions, PostprocessResult, SoftNmsMethod,
    SoftNmsOptions,
};
/// Error returned by crop operations.
pub use crop::{center_crop_image, crop_image, CropError, CropInfo, CropResult};
/// Error returned by tensor layout operations.
pub use layout::{chw_to_hwc, hwc_to_chw, nchw_to_nhwc, nhwc_to_nchw, rgb_to_bgr, LayoutError};
/// Error returned by letterbox operations.
pub use letterbox::{
    compute_letterbox, letterbox_image, LetterboxError, LetterboxInfo, LetterboxResult, Padding,
};
/// Error returned by segmentation mask operations.
pub use mask::{
    letterbox_mask, mask_to_box, resize_mask, threshold_mask, unletterbox_mask,
    LetterboxMaskResult, MaskError, ResizeMaskResult,
};
/// Error returned by normalization operations.
pub use normalize::{normalize_image, NormalizeError, NormalizeInfo, NormalizeResult};
/// Error returned by fused preprocessing operations.
pub use preprocess::{
    preprocess_batch, preprocess_image, PreprocessBatchResult, PreprocessError, PreprocessGeometry,
    PreprocessInfo, PreprocessLayout, PreprocessMode, PreprocessResult,
};
/// Error returned by direct resize operations.
pub use resize::{resize_image, ResizeError, ResizeInfo, ResizeResult};
