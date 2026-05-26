# Rust API

`rusty-cv` exposes a small set of image and box-processing helpers intended to compose cleanly in inference pipelines.

## Image operations

### `resize_image`

Resizes an image directly to the requested width and height.

- aspect ratio is not preserved
- useful for deterministic fixed-size model inputs
- returns a `ResizeResult` with the output image and metadata

```rust
use image::imageops::FilterType;
use rusty_cv::resize_image;

let image = image::open("input.jpg")?;
let result = resize_image(&image, 320, 240, FilterType::Triangle)?;
result.image.save("resized.jpg")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### `compute_letterbox`

Computes the geometry of an aspect-ratio-preserving resize without touching pixel data.

- returns scale
- returns resized dimensions
- returns padding split across all four sides

This is useful when you want resize metadata for later coordinate remapping or parity checks against another implementation.

### `letterbox_image`

Resizes an image to fit inside a target frame and fills the remaining area with a color.

- preserves aspect ratio
- centers the resized content
- returns the output image and full geometry metadata

```rust
use image::imageops::FilterType;
use rusty_cv::letterbox_image;

let image = image::open("input.jpg")?;
let result = letterbox_image(&image, 640, 640, [114, 114, 114], FilterType::Triangle)?;
println!("{:?}", result.info);
result.image.save("letterboxed.jpg")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### `crop_image`

Crops an explicit rectangle from the source image.

- input is `x`, `y`, `width`, `height`
- returns the cropped image and the resolved crop metadata

### `center_crop_image`

Crops a centered region from the source image.

- useful for common classification-style preprocessing
- returns the cropped image and the derived crop metadata

### `normalize_image`

Converts RGB pixels into a contiguous `f32` HWC buffer.

- configurable per-channel `mean`
- configurable per-channel `std`
- optional `0..255 -> 0.0..1.0` scaling

```rust
use rusty_cv::normalize_image;

let image = image::open("input.jpg")?;
let result = normalize_image(
    &image,
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
    true,
)?;
println!("{}x{}", result.info.width, result.info.height);
println!("{}", result.data.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

### `preprocess_image`

Fuses resize or letterbox geometry with normalization and tensor layout
conversion.

- supports `PreprocessMode::Resize`
- supports `PreprocessMode::Letterbox { fill }`
- supports `PreprocessLayout::Hwc` and `PreprocessLayout::Chw`

```rust
use image::imageops::FilterType;
use rusty_cv::{preprocess_image, PreprocessLayout, PreprocessMode};

let image = image::open("input.jpg")?;
let result = preprocess_image(
    &image,
    640,
    640,
    PreprocessMode::Letterbox { fill: [114, 114, 114] },
    FilterType::Triangle,
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
    true,
    PreprocessLayout::Chw,
)?;
println!("{:?}", result.info);
println!("{}", result.data.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Box operations

### `BBoxXYXY`

Axis-aligned box type using `x1, y1, x2, y2` coordinates.

### `BBoxXYWH`

Axis-aligned box type using `x, y, width, height` coordinates.

### `iou`

Computes intersection-over-union for two `BBoxXYXY` values.

### Box conversion and remapping

The crate also exposes:

- `xyxy_to_xywh`
- `xywh_to_xyxy`
- `clip_boxes`
- `resize_boxes`
- `letterbox_boxes`
- `unletterbox_boxes`

### `nms`

Runs single-class non-maximum suppression over boxes and scores.

- expects one score per box
- returns indices of boxes to keep
- useful after detector inference

```rust
use rusty_cv::{BBoxXYXY, nms};

let boxes = [
    BBoxXYXY { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 },
    BBoxXYXY { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0 },
    BBoxXYXY { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0 },
];
let scores = [0.9, 0.8, 0.7];

let keep = nms(&boxes, &scores, 0.5)?;
println!("{keep:?}");
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Public modules

- `rusty_cv::bbox`
- `rusty_cv::crop`
- `rusty_cv::letterbox`
- `rusty_cv::normalize`
- `rusty_cv::preprocess`
- `rusty_cv::resize`
