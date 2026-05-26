# Detection Postprocessing

`rusty-cv` includes a small postprocessing stack aimed at object-detection style inference pipelines in Rust and Python.

The functions in this guide are the “harder” CV pieces in the crate today. They are also the functions most aligned with the library goal of moving slow Python-side postprocessing into Rust.

## Function families

### Hard NMS

- `nms(...)`
- `batched_nms(...)`
- `multiclass_nms(...)`

Use hard NMS when overlapping boxes should be fully removed once they exceed the IoU threshold.

### Soft NMS

- `soft_nms(...)`
- `batched_soft_nms(...)`
- `multiclass_soft_nms(...)`

Use soft NMS when overlapping boxes should stay alive with decayed scores instead of being removed outright.

## Input shapes

### Single-class

`nms(...)` and `soft_nms(...)` operate on:

- `boxes`: `N x 4`
- `scores`: `N`

### Batched / class-aware

`batched_nms(...)` and `batched_soft_nms(...)` operate on:

- `boxes`: `N x 4`
- `scores`: `N`
- `class_ids`: `N`

Only boxes from the same class interact with one another.

### Multiclass

`multiclass_nms(...)` and `multiclass_soft_nms(...)` operate on:

- `boxes`: `N x 4`
- `class_scores`: `N x C`

Each class is expanded into its own candidate stream, processed independently, then merged back into one globally sorted detection list.

## Options

### `NmsOptions`

Hard NMS uses:

- `iou_threshold`
- `score_threshold`
- `pre_nms_top_k`
- `max_detections`

### `SoftNmsOptions`

Soft NMS uses:

- `method`
- `iou_threshold`
- `score_threshold`
- `sigma`
- `pre_nms_top_k`
- `max_detections`

`method` can be:

- `Linear`
- `Gaussian`

Behavior:

- `Linear`: scores are reduced only when IoU is above the threshold
- `Gaussian`: scores are always decayed as a smooth function of overlap

## Return format

Rust advanced APIs return `Vec<Detection>` where each item contains:

- `box_index`
- `class_id`
- `score`

Python bindings expose the same information as dictionaries of NumPy arrays:

- `indices`
- `class_ids`
- `scores`

## Ordering rules

- outputs are sorted by score descending
- ties fall back to the lower original box index
- merged multiclass outputs use class id as a later tie-break

## Rust examples

### Hard NMS

```rust
use rusty_cv::{batched_nms, multiclass_nms, nms, BBoxXYXY, NmsOptions};

let boxes = [
    BBoxXYXY { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 },
    BBoxXYXY { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0 },
    BBoxXYXY { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0 },
];
let scores = [0.9, 0.8, 0.7];
let class_ids = [0usize, 0usize, 1usize];
let class_scores = [
    0.9, 0.1,
    0.8, 0.75,
    0.1, 0.7,
];

let keep = nms(&boxes, &scores, 0.5)?;
let batched = batched_nms(&boxes, &scores, &class_ids, &NmsOptions::default())?;
let multi = multiclass_nms(&boxes, &class_scores, 2, &NmsOptions::default())?;
# let _ = (keep, batched, multi);
# Ok::<(), rusty_cv::BBoxError>(())
```

### Soft NMS

```rust
use rusty_cv::{
    batched_soft_nms, multiclass_soft_nms, soft_nms, BBoxXYXY, SoftNmsMethod, SoftNmsOptions,
};

let boxes = [
    BBoxXYXY { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 },
    BBoxXYXY { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0 },
    BBoxXYXY { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0 },
];
let scores = [0.9, 0.8, 0.7];
let class_ids = [0usize, 0usize, 1usize];
let class_scores = [
    0.9, 0.1,
    0.8, 0.75,
    0.1, 0.7,
];

let options = SoftNmsOptions {
    method: SoftNmsMethod::Linear,
    iou_threshold: 0.5,
    score_threshold: 0.25,
    sigma: 0.5,
    pre_nms_top_k: None,
    max_detections: Some(4),
};

let soft = soft_nms(&boxes, &scores, &options)?;
let batched_soft = batched_soft_nms(&boxes, &scores, &class_ids, &options)?;
let multi_soft = multiclass_soft_nms(&boxes, &class_scores, 2, &options)?;
# let _ = (soft, batched_soft, multi_soft);
# Ok::<(), rusty_cv::BBoxError>(())
```

## Python examples

### Hard NMS

```python
import numpy as np
import rusty_cv

boxes = np.array(
    [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
    ],
    dtype=np.float32,
)
scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
class_ids = np.array([0, 0, 1], dtype=np.int64)
class_scores = np.array(
    [
        [0.9, 0.1],
        [0.8, 0.75],
        [0.1, 0.7],
    ],
    dtype=np.float32,
)

keep = rusty_cv.nms(boxes, scores, iou_threshold=0.5)
batched = rusty_cv.batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
multi = rusty_cv.multiclass_nms(
    boxes,
    class_scores,
    iou_threshold=0.5,
    score_threshold=0.7,
    max_detections=3,
)
```

### Soft NMS

```python
soft = rusty_cv.soft_nms(
    boxes,
    scores,
    method="linear",
    iou_threshold=0.5,
    score_threshold=0.25,
)

soft_batched = rusty_cv.batched_soft_nms(
    boxes,
    scores,
    class_ids,
    method="linear",
    iou_threshold=0.5,
    score_threshold=0.25,
)

soft_multi = rusty_cv.multiclass_soft_nms(
    boxes,
    class_scores,
    method="linear",
    iou_threshold=0.5,
    score_threshold=0.25,
    max_detections=4,
)
```

## Parity and timing

Reference script:

```bash
python3 scripts/nms_ref.py
```

Comparison script:

```bash
.venv/bin/python scripts/bench_nms.py 64 20 0.5 4 0.25
```

The benchmark script checks parity and measures:

- single hard NMS
- batched hard NMS
- multiclass hard NMS
- single soft NMS
- batched soft NMS
- multiclass soft NMS

Example output observed in this repository:

```text
[single]
rust_us_per_iter=114.01
python_us_per_iter=921.99
speedup=8.09x
[batched]
rust_us_per_iter=81.77
python_us_per_iter=233.61
speedup=2.86x
[multiclass]
rust_us_per_iter=368.89
python_us_per_iter=3389.74
speedup=9.19x
[soft]
rust_us_per_iter=129.45
python_us_per_iter=1116.96
speedup=8.63x
[soft_batched]
rust_us_per_iter=68.73
python_us_per_iter=317.48
speedup=4.62x
[soft_multiclass]
rust_us_per_iter=483.40
python_us_per_iter=4697.54
speedup=9.72x
```

These are local measurements on this machine, not portability claims.
