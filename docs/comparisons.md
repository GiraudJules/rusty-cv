# Comparisons

This repository includes small parity and timing scripts to compare the Rust implementation with simple Python references.

These scripts are useful for two different purposes:

- checking behavioral parity while the APIs are still evolving
- getting rough local timing comparisons on the same machine

## Letterbox comparison

The repository includes:

- `scripts/letterbox_ref.py`
- `examples/bench_letterbox.rs`
- `scripts/bench_letterbox.py`

Geometry parity example:

```bash
python3 scripts/letterbox_ref.py \
  --source-width 1920 \
  --source-height 1080 \
  --target-width 640 \
  --target-height 640
```

Timing comparison example:

```bash
cargo run --release --example bench_letterbox -- 2000000
python3 scripts/bench_letterbox.py 2000000
```

This benchmark only measures the resize geometry calculation, not full image resampling.

## NMS comparison

The repository includes:

- `scripts/nms_ref.py`
- `scripts/bench_nms.py`

Parity check:

```bash
python3 scripts/nms_ref.py
```

Rust vs Python timing:

```bash
.venv/bin/python scripts/bench_nms.py 256 500 0.5
```

The benchmark script:

- generates the same synthetic boxes and scores for both implementations
- checks that Rust and Python return the same kept indices
- reports per-iteration timings for each implementation

Example output observed in this repository:

```text
boxes=256
iterations=500
iou_threshold=0.5
rust_us_per_iter=704.01
python_us_per_iter=7792.78
speedup=11.07x
```

## Benchmark caveats

- these are local machine measurements, not portable claims
- image encode/decode overhead is not represented in the NumPy-oriented comparisons
- algorithmic changes can matter more than language choice for larger workloads
