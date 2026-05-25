#!/usr/bin/env python3

from __future__ import annotations

import sys
import time

from letterbox_ref import compute_letterbox


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    checksum = 0

    start = time.perf_counter()

    for i in range(iterations):
        original_width = 320 + (i % 1600)
        original_height = 240 + (i % 1200)
        target_width = 640 + (i % 2) * 640
        target_height = 640 + ((i // 2) % 2) * 640

        info = compute_letterbox(
            original_width,
            original_height,
            target_width,
            target_height,
        )

        checksum += (
            info["resized_width"]
            + info["resized_height"]
            + info["padding"]["left"]
            + info["padding"]["top"]
        )

    elapsed = time.perf_counter() - start

    print(f"iterations={iterations}")
    print(f"elapsed_s={elapsed:.6f}")
    print(f"ns_per_iter={(elapsed * 1e9) / iterations:.2f}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
