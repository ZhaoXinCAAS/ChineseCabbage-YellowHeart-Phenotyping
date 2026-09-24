"""Color calibration primitives (Reviewer-2 Q2, Reviewer-1 Q6).

White-balance / illumination correction applied *before* segmentation and
feature extraction, so Figure 1 can state a calibration step instead of
"no calibration". Two paths:

* `grayworld` — gray-world white balance (no chart needed); scales B/G/R
  channel means to the global mean. Documents the acquisition-side fix.
* `--ccm CSV` — explicit 3x3 color-correction matrix measured from a color
  chart in the shooting rig (CSV, rows R/G/B gains in BGR column order);
  applied as `out = CCM @ in` per pixel.

Also hosts the image-perturbation operators used by `robustness_test.py`
to simulate illumination drift (brightness / color-temperature shift).

Usage
-----
.. code-block:: bash
    uv run color-calibrate --input raw.jpg --output wb.jpg --method grayworld
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

EPS = 1e-9


def gray_world_wb(img_bgr: np.ndarray) -> np.ndarray:
    """Scale B/G/R means toward the global mean; uint8 in, uint8 out."""
    img = img_bgr.astype(np.float64)
    means = img.reshape(-1, 3).mean(axis=0)
    global_mean = means.mean()
    gains = global_mean / np.clip(means, EPS, None)
    out = np.clip(img * gains.reshape(1, 1, 3), 0, 255)
    return out.astype(np.uint8)


def apply_ccm(img_bgr: np.ndarray, ccm: np.ndarray):
    """Apply a 3x3 correction matrix (BGR-ordered) to a BGR image."""
    mat = np.asarray(ccm, dtype=np.float64).reshape(3, 3)
    out = np.clip(img_bgr.astype(np.float64) @ mat.T, 0, 255)
    return out.astype(np.uint8)


def load_ccm(path: str) -> npt.NDArray[np.float64]:
    """Load a 3x3 color-correction matrix from a CSV file."""
    raw = np.asarray(np.loadtxt(path, delimiter=",", dtype=float))
    assert isinstance(raw, np.ndarray)
    if raw.size != 9:
        raise ValueError(f"CCM CSV must hold 9 values, got {raw.size}: {path}")
    return raw.reshape(3, 3)


def apply_brightness(img_bgr: np.ndarray, factor: float):
    """Simulate exposure drift by scaling the HSV Value channel."""
    if factor == 1.0:
        return img_bgr.copy()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_temp_shift(img_bgr: np.ndarray, gain: float):
    """Simulate white-balance drift: R *= gain, B /= gain (approximate)."""
    if gain == 1.0:
        return img_bgr.copy()
    out = img_bgr.astype(np.float64)
    out[:, :, 2] = np.clip(out[:, :, 2] * gain, 0, 255)  # R
    out[:, :, 0] = np.clip(out[:, :, 0] / max(gain, EPS), 0, 255)  # B
    return out.astype(np.uint8)


def calibrate_file(
    input_path: str, output_path: str, method: str = "grayworld", ccm: np.ndarray | None = None
):
    """White-balance (and optionally CCM-correct) one image file."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")
    if method == "grayworld":
        img = gray_world_wb(img)
    if ccm is not None:
        img = apply_ccm(img, ccm)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)


@dataclass(frozen=True, slots=True)
class ColorCalibrateArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    method: str
    ccm: str | None


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="White-balance / CCM calibration.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["none", "grayworld"], default="grayworld")
    p.add_argument("--ccm", default=None, help="3x3 CCM CSV (optional).")
    return cast(ColorCalibrateArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    try:
        ccm = load_ccm(args.ccm) if args.ccm else None
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    try:
        calibrate_file(args.input, args.output, args.method, ccm)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Calibrated image saved to: {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
