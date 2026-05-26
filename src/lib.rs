//! `rusty-cv` is a small Rust-first computer vision crate focused on simple,
//! reusable image preprocessing primitives.
//!
//! Current functionality:
//!
//! - box geometry helpers, IoU, hard NMS variants, and soft NMS variants
//! - direct crop and center crop
//! - direct resize to exact dimensions
//! - letterbox resize with aspect-ratio preservation and padding
//! - RGB normalization into contiguous `f32` buffers
//! - optional Python bindings behind the `python` feature

/// Bounding-box geometry and postprocessing operations.
pub mod bbox;
/// Crop geometry and image operations.
pub mod crop;
/// Letterbox geometry and image operations.
pub mod letterbox;
/// Image normalization operations.
pub mod normalize;
/// Direct image resize operations.
pub mod resize;

#[cfg(feature = "python")]
mod python;

/// Error returned by box postprocessing operations.
pub use bbox::{
    batched_nms, batched_soft_nms, iou, multiclass_nms, multiclass_soft_nms, nms, nms_with_options,
    soft_nms, BBoxError, BBoxXYXY, Detection, NmsOptions, SoftNmsMethod, SoftNmsOptions,
};
/// Error returned by crop operations.
pub use crop::{center_crop_image, crop_image, CropError, CropInfo, CropResult};
/// Error returned by letterbox operations.
pub use letterbox::{
    compute_letterbox, letterbox_image, LetterboxError, LetterboxInfo, LetterboxResult, Padding,
};
/// Error returned by normalization operations.
pub use normalize::{normalize_image, NormalizeError, NormalizeInfo, NormalizeResult};
/// Error returned by direct resize operations.
pub use resize::{resize_image, ResizeError, ResizeInfo, ResizeResult};
