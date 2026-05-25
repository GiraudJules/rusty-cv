# rusty-cv

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

`rusty-cv` is a small computer vision library for Rust with optional Python bindings built from the same core implementation.

It currently focuses on the practical preprocessing and postprocessing steps that show up in detection and inference pipelines:

- resize
- letterbox resize
- crop and center crop
- RGB normalization
- bounding box IoU
- non-maximum suppression
- optional NumPy-backed Python bindings

## Why this crate

- Rust-first core with a small, explicit API surface
- Python extension module generated from the same Rust codebase
- Useful building blocks for inference preprocessing and detection postprocessing
- No OpenCV dependency

## Installation

### Rust

```toml
[dependencies]
rusty-cv = "0.1.0"
image = "0.25"
```

For local development:

```toml
[dependencies]
rusty-cv = { path = "../rusty-cv" }
image = "0.25"
```

### Python

The Python module is built with `maturin` and exposed as `rusty_cv`.

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop --features python
```

## Rust quick start

```rust
use image::imageops::FilterType;
use rusty_cv::{BBoxXYXY, center_crop_image, iou, letterbox_image, nms, normalize_image, resize_image};

let image = image::open("input.jpg")?;
let cropped = center_crop_image(&image, 224, 224)?;
let resized = resize_image(&image, 320, 240, FilterType::Triangle)?;
let letterboxed = letterbox_image(&image, 640, 640, [114, 114, 114], FilterType::Triangle)?;
let normalized = normalize_image(&image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], true)?;

let boxes = [
    BBoxXYXY { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 },
    BBoxXYXY { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0 },
];
let scores = [0.9, 0.8];

cropped.image.save("cropped.jpg")?;
resized.image.save("resized.jpg")?;
letterboxed.image.save("letterboxed.jpg")?;
println!("{}", normalized.data.len());
println!("{}", iou(boxes[0], boxes[1]));
println!("{:?}", nms(&boxes, &scores, 0.5)?);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Python quick start

### Byte-based API

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

Path("resized.png").write_bytes(resized)
Path("letterboxed.png").write_bytes(letterboxed)
```

### NumPy API

```python
import numpy as np
import rusty_cv

image = np.zeros((480, 640, 3), dtype=np.uint8)

letterboxed, info = rusty_cv.letterbox_image_numpy(
    image,
    640,
    640,
    fill=(114, 114, 114),
    filter="triangle",
)

normalized = rusty_cv.normalize_image_numpy(
    image,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    scale_to_unit=True,
)

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
print(letterboxed.shape, info)
print(normalized.dtype, normalized.shape)
print(keep)
```

## Package status

`rusty-cv` is usable today for small CV preprocessing and postprocessing tasks, but the API is still early and intentionally compact. The current Python NumPy entrypoints support `H x W x 3` `uint8` inputs, and the crate is still growing toward a broader public release.

## Documentation

- [docs/README.md](docs/README.md)
- [docs/getting-started.md](docs/getting-started.md)
- [docs/rust-api.md](docs/rust-api.md)
- [docs/python-api.md](docs/python-api.md)
- [docs/comparisons.md](docs/comparisons.md)
