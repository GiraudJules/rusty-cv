# rusty-cv

Small Rust-first computer vision primitives with optional Python bindings.

`rusty-cv` is intended to become a lightweight public library for common image preprocessing tasks that are useful in both Rust applications and Python pipelines.

Current scope:

- IoU and non-maximum suppression
- crop
- center crop
- exact resize
- letterbox resize
- RGB normalization
- Rust API
- optional Python extension module via `pyo3` + `maturin`

The core implementations live in [src/bbox.rs](src/bbox.rs), [src/crop.rs](src/crop.rs), [src/resize.rs](src/resize.rs), [src/letterbox.rs](src/letterbox.rs), and [src/normalize.rs](src/normalize.rs).

## Features

### `resize_image(...)`

Resize directly to the requested width and height.

- does not preserve aspect ratio
- useful for deterministic model input resizing

### `compute_letterbox(...)`

Compute the geometry of an aspect-ratio-preserving resize without touching image pixels.

- returns scale
- returns resized dimensions
- returns symmetric padding split

### `letterbox_image(...)`

Resize to fit within a target frame and fill the remaining area with a padding color.

- preserves aspect ratio
- centers the resized image
- returns image + geometry metadata

### `crop_image(...)` and `center_crop_image(...)`

Crop exact or centered regions from an image.

- `crop_image(...)` uses an explicit rectangle
- `center_crop_image(...)` derives the rectangle from the source center

### `normalize_image(...)`

Normalize RGB pixels into a contiguous `f32` HWC buffer.

- configurable `mean` and `std`
- optional `0..255 -> 0.0..1.0` scaling
- useful before model inference

### `iou(...)` and `nms(...)`

Bounding-box postprocessing helpers for detection pipelines.

- `iou(...)` computes box overlap in `xyxy` format
- `nms(...)` performs single-class non-maximum suppression
- Python bindings accept `Nx4` `float32` boxes and `N` `float32` scores

## Rust installation

Add the crate to `Cargo.toml`:

```toml
[dependencies]
rusty-cv = "0.1.0"
image = "0.25"
```

For a local checkout:

```toml
[dependencies]
rusty-cv = { path = "../rusty-cv" }
image = "0.25"
```

## Rust usage

```rust
use image::imageops::FilterType;
use rusty_cv::{BBoxXYXY, center_crop_image, iou, letterbox_image, nms, normalize_image, resize_image};

let image = image::open("input.jpg")?;
let cropped = center_crop_image(&image, 224, 224)?;
let resized = resize_image(&image, 320, 240, FilterType::Triangle)?;
let result = letterbox_image(&image, 640, 640, [114, 114, 114], FilterType::Triangle)?;
let normalized = normalize_image(&image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], true)?;
let boxes = [
    BBoxXYXY { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 },
    BBoxXYXY { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0 },
];
let scores = [0.9, 0.8];

cropped.image.save("cropped.jpg")?;
resized.image.save("resized.jpg")?;
println!("{:?}", result.info);
result.image.save("output.jpg")?;
println!("{}", normalized.data.len());
println!("{}", iou(boxes[0], boxes[1]));
println!("{:?}", nms(&boxes, &scores, 0.5)?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`resize_image(...)` resizes directly to the requested width and height.
`compute_letterbox(...)` returns just the geometry, which is useful if you want to compare padding and scale against another implementation.

## Python installation

The crate includes an optional Python extension module powered by `pyo3` and `maturin`.

Build and install it into your active Python environment with:

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop --features python
```

This produces an importable `rusty_cv` Python module from the same Rust codebase.

## Python usage

The current Python API is byte-oriented:

- input: encoded image bytes such as PNG or JPEG
- output: encoded image bytes
- metadata: regular Python dictionaries

Example:

```python
from pathlib import Path
import rusty_cv

input_bytes = Path("input.jpg").read_bytes()

resized = rusty_cv.resize_image(
    input_bytes,
    320,
    240,
    filter="triangle",
    output_format="png",
)

letterboxed = rusty_cv.letterbox_image(
    input_bytes,
    640,
    640,
    fill=(114, 114, 114),
    filter="triangle",
    output_format="png",
)

info = rusty_cv.compute_letterbox(1920, 1080, 640, 640)

Path("resized.png").write_bytes(resized)
Path("letterboxed.png").write_bytes(letterboxed)
print(info)
```

For NumPy-based CV pipelines, the module also exposes array-first entrypoints for `H x W x 3` `uint8` arrays:

```python
import numpy as np
import rusty_cv

image = np.zeros((480, 640, 3), dtype=np.uint8)

resized = rusty_cv.resize_image_numpy(image, 320, 320, filter="triangle")
letterboxed, info = rusty_cv.letterbox_image_numpy(
    image,
    640,
    640,
    fill=(114, 114, 114),
    filter="triangle",
)

print(resized.shape)
print(letterboxed.shape, info)
```

Normalization is exposed as a NumPy-first API and returns `float32`:

```python
normalized = rusty_cv.normalize_image_numpy(
    image,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    scale_to_unit=True,
)

print(normalized.dtype, normalized.shape)
```

NMS is also exposed in the Python extension:

```python
boxes = np.array(
    [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
    ],
    dtype=np.float32,
)
scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

keep = rusty_cv.nms(boxes, scores, iou_threshold=0.5)
print(keep)
```

Current NumPy constraints:

- input dtype must be `uint8`
- input shape must be `H x W x 3`
- output is also `H x W x 3` `uint8`

## Python reference script

The reference script mirrors the same math and can also render an output image when Pillow is installed:

```bash
python3 scripts/letterbox_ref.py --source-width 1920 --source-height 1080 --target-width 640 --target-height 640
python3 scripts/letterbox_ref.py --input input.jpg --output output.jpg --target-width 640 --target-height 640
```

For a simple Rust/Python timing comparison of the geometry calculation only:

```bash
cargo run --release --example bench_letterbox -- 2000000
python3 scripts/bench_letterbox.py 2000000
```

For NMS, the repo includes a pure Python reference and a Rust-vs-Python comparison script:

```bash
python3 scripts/nms_ref.py
.venv/bin/python scripts/bench_nms.py 256 500 0.5
```

The benchmark script first checks that the Rust and Python implementations return identical kept indices, then reports elapsed time for both implementations on the same synthetic inputs.

Example result from this repository on the current development machine:

```text
boxes=256
iterations=500
iou_threshold=0.5
rust_us_per_iter=746.51
python_us_per_iter=8023.85
speedup=10.75x
```

## Project Status

`rusty-cv` is usable today for basic resize and letterbox preprocessing in both Rust and Python, but it is still early-stage.

Current expectations:

- the API surface is intentionally small
- Python support is available and now includes byte-based and NumPy-based entrypoints
- the project is still evolving toward a more complete public release
- some packaging and release automation work is still ahead

## Verification

- `cargo test` currently passes in this environment.
- `cargo check --features python` currently passes in this environment.
- `maturin develop --features python` succeeds in a local virtual environment.
- `scripts/python_smoke_test.py` passes against the built extension module.
