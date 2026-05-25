//! `rusty-cv` is a small Rust-first computer vision crate focused on simple,
//! reusable image preprocessing primitives.
//!
//! Current functionality:
//!
//! - box geometry helpers, IoU, and NMS
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
pub use bbox::{BBoxError, BBoxXYXY, iou, nms};
/// Error returned by crop operations.
pub use crop::{CropError, CropInfo, CropResult, center_crop_image, crop_image};
/// Error returned by letterbox operations.
pub use letterbox::{
    LetterboxError, LetterboxInfo, LetterboxResult, Padding, compute_letterbox,
    letterbox_image,
};
/// Error returned by normalization operations.
pub use normalize::{NormalizeError, NormalizeInfo, NormalizeResult, normalize_image};
/// Error returned by direct resize operations.
pub use resize::{ResizeError, ResizeInfo, ResizeResult, resize_image};
