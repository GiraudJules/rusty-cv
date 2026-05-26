# rusty-cv

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

Small Rust-first computer vision primitives with optional Python bindings built from the same core implementation.

`rusty-cv` is aimed at two practical use cases:

- Rust applications that need lightweight CV preprocessing or detection postprocessing helpers
- Python CV / deep learning pipelines that want expensive steps moved out of Python

Current scope:

- exact resize
- letterbox resize
- crop and center crop
- RGB normalization
- fused preprocess for tensor-ready HWC/CHW output
- IoU
- box conversion, clipping, and geometry remapping helpers
- hard NMS: single-class, batched, multiclass
- soft NMS: single-class, batched, multiclass
- optional Python extension module via `pyo3` + `maturin`

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

This builds an importable `rusty_cv` module from the same Rust crate.

## Rust quick start

```rust
use image::imageops::FilterType;
use rusty_cv::{letterbox_image, nms, resize_image, BBoxXYXY};

let image = image::open("input.jpg")?;
let resized = resize_image(&image, 320, 240, FilterType::Triangle)?;
let letterboxed = letterbox_image(&image, 640, 640, [114, 114, 114], FilterType::Triangle)?;

let boxes = [
    BBoxXYXY { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 },
    BBoxXYXY { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0 },
    BBoxXYXY { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0 },
];
let scores = [0.9, 0.8, 0.7];

resized.image.save("resized.jpg")?;
letterboxed.image.save("letterboxed.jpg")?;
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

tensor, info = rusty_cv.preprocess_image_numpy(
    image,
    640,
    640,
    mode="letterbox",
    fill=(114, 114, 114),
    filter="triangle",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    scale_to_unit=True,
    layout="chw",
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
mapped = rusty_cv.letterbox_boxes_numpy(boxes, 640, 480, 640, 640)
print(tensor.shape, info)
print(mapped)
print(keep)
```

## Detection Postprocessing

The library now includes:

- `nms(...)`
- `batched_nms(...)`
- `multiclass_nms(...)`
- `soft_nms(...)`
- `batched_soft_nms(...)`
- `multiclass_soft_nms(...)`

The detailed documentation for these APIs, including behavior, return shapes, and local Rust/Python timing comparisons, lives in [docs/postprocessing.md](docs/postprocessing.md).

## Documentation

- [docs/README.md](docs/README.md)
- [docs/getting-started.md](docs/getting-started.md)
- [docs/rust-api.md](docs/rust-api.md)
- [docs/python-api.md](docs/python-api.md)
- [docs/comparisons.md](docs/comparisons.md)
- [docs/postprocessing.md](docs/postprocessing.md)

## Project Status

`rusty-cv` is usable today for small CV preprocessing and detection postprocessing tasks in both Rust and Python, but the crate is still early and the public API is intentionally compact.

## License

No license file has been added yet. Until one is present, this repository should not be treated as open source.
