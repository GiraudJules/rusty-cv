# Getting Started

This repository is organized as a Rust library first, with optional Python bindings generated from the same implementation.

## What `rusty-cv` currently includes

- exact resize
- aspect-ratio-preserving letterbox resize
- crop and center crop
- RGB normalization into `f32`
- bounding box IoU
- single-class NMS
- Python bindings for byte-oriented and NumPy-oriented workflows

## Rust setup

Add the crate and the `image` crate to your project:

```toml
[dependencies]
rusty-cv = "0.1.0"
image = "0.25"
```

During local development against this repository:

```toml
[dependencies]
rusty-cv = { path = "../rusty-cv" }
image = "0.25"
```

Then import the operations you need:

```rust
use image::imageops::FilterType;
use rusty_cv::{compute_letterbox, letterbox_image, resize_image};
```

## Python setup

The Python extension module is built from the Rust crate with `maturin`.

```bash
python3 -m venv .venv
.venv/bin/pip install maturin numpy
.venv/bin/maturin develop --features python
```

This installs an importable `rusty_cv` module into the active environment.

## Build and verification

Rust-only validation:

```bash
cargo test
```

Rust + Python binding validation:

```bash
cargo check --features python
.venv/bin/maturin develop --features python
.venv/bin/python scripts/python_smoke_test.py
```

## Current constraints

- Python NumPy image APIs accept `H x W x 3` `uint8` arrays
- normalization currently returns `H x W x 3` `float32`
- NMS is single-class and returns kept indices
- the crate currently targets practical inference utilities, not full augmentation pipelines

## Next docs

- [rust-api.md](rust-api.md)
- [python-api.md](python-api.md)
- [comparisons.md](comparisons.md)
