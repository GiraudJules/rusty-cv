#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None


def compute_letterbox(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> dict:
    if original_width <= 0 or original_height <= 0:
        raise ValueError("source image width and height must be greater than zero")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target width and height must be greater than zero")

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    pad_x = target_width - resized_width
    pad_y = target_height - resized_height

    return {
        "original_width": original_width,
        "original_height": original_height,
        "target_width": target_width,
        "target_height": target_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale": scale,
        "padding": {
            "left": pad_x // 2,
            "right": pad_x - (pad_x // 2),
            "top": pad_y // 2,
            "bottom": pad_y - (pad_y // 2),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference Python letterbox implementation matching src/letterbox.rs",
    )
    parser.add_argument("--input", type=Path, help="Optional input image path")
    parser.add_argument("--output", type=Path, help="Optional output image path")
    parser.add_argument("--source-width", type=int, help="Width for geometry-only comparison")
    parser.add_argument("--source-height", type=int, help="Height for geometry-only comparison")
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument(
        "--fill",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(114, 114, 114),
        help="Padding color",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        if Image is None:
            raise SystemExit("Pillow is required when --input is used. Install it with: python3 -m pip install pillow")
        with Image.open(args.input) as image:
            width, height = image.size
            info = compute_letterbox(width, height, args.target_width, args.target_height)
            resized = image.convert("RGB").resize(
                (info["resized_width"], info["resized_height"]),
                Image.Resampling.BILINEAR,
            )
            canvas = Image.new("RGB", (args.target_width, args.target_height), tuple(args.fill))
            canvas.paste(
                resized,
                (info["padding"]["left"], info["padding"]["top"]),
            )
            if args.output:
                canvas.save(args.output)
    else:
        if args.source_width is None or args.source_height is None:
            raise SystemExit("Provide either --input or both --source-width and --source-height")
        info = compute_letterbox(
            args.source_width,
            args.source_height,
            args.target_width,
            args.target_height,
        )

    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
