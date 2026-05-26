# Python API

The Python module is named `rusty_cv` and is built from the same Rust implementation as the crate.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install maturin numpy
.venv/bin/maturin develop --features python
```

## API styles

There are two Python API styles in the module:

- byte-oriented image APIs for encoded PNG/JPEG buffers
- NumPy-oriented APIs for `H x W x 3` `uint8` arrays

## Byte-oriented image APIs

These functions decode image bytes, run the operation, then encode the result back to bytes.

### `resize_image`

```python
output = rusty_cv.resize_image(
    input_bytes,
    320,
    240,
    filter="triangle",
    output_format="png",
)
```

### `letterbox_image`

```python
output = rusty_cv.letterbox_image(
    input_bytes,
    640,
    640,
    fill=(114, 114, 114),
    filter="triangle",
    output_format="png",
)
```

### `crop_image`

```python
output = rusty_cv.crop_image(
    input_bytes,
    10,
    20,
    224,
    224,
    output_format="png",
)
```

### `center_crop_image`

```python
output = rusty_cv.center_crop_image(
    input_bytes,
    224,
    224,
    output_format="png",
)
```

### `compute_letterbox`

Returns a dictionary containing:

- `original_width`
- `original_height`
- `target_width`
- `target_height`
- `resized_width`
- `resized_height`
- `scale`
- `padding`

The `padding` value is a nested dict with `top`, `bottom`, `left`, and `right`.

## NumPy image APIs

These APIs accept `H x W x 3` `uint8` arrays and avoid encode/decode overhead.

### `resize_image_numpy`

```python
resized = rusty_cv.resize_image_numpy(image, 320, 240, filter="triangle")
```

### `letterbox_image_numpy`

Returns `(image, info)`.

```python
letterboxed, info = rusty_cv.letterbox_image_numpy(
    image,
    640,
    640,
    fill=(114, 114, 114),
    filter="triangle",
)
```

### `crop_image_numpy`

Returns `(image, info)`.

```python
cropped, info = rusty_cv.crop_image_numpy(image, 10, 20, 224, 224)
```

### `center_crop_image_numpy`

Returns `(image, info)`.

```python
cropped, info = rusty_cv.center_crop_image_numpy(image, 224, 224)
```

### `normalize_image_numpy`

Returns a `float32` `H x W x 3` array.

```python
normalized = rusty_cv.normalize_image_numpy(
    image,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    scale_to_unit=True,
)
```

## Box APIs

### `iou`

Accepts two 4-tuples in `xyxy` format and returns a float:

```python
score = rusty_cv.iou((0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0))
```

### `nms`

Accepts:

- `boxes`: `N x 4` `float32`
- `scores`: `N` `float32`
- `iou_threshold`: `float`

Returns a Python list of kept indices.

```python
keep = rusty_cv.nms(boxes, scores, iou_threshold=0.5)
```

## Accepted filter names

- `nearest`
- `triangle`
- `bilinear`
- `catmull_rom`
- `catmull-rom`
- `gaussian`
- `lanczos3`
- `lanczos`

## Current constraints

- NumPy image inputs must be `uint8`
- NumPy image inputs must be shaped `H x W x 3`
- grayscale, RGBA, CHW, and batched arrays are not supported yet
