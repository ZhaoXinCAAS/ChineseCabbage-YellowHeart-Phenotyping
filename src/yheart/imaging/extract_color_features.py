"""Extract quantitative yellow-heart phenotypic color traits.

From each cropped cross-section image plus its JSON (containing the
`Pan_center_contour_area` head mask and the `Yellow_area` segmented mask),
this script computes the yellow-heart area ratio and ten multidimensional
color-space traits (R/G/B, HSV H circular mean + S/V, CIELAB L/a*/b*, ExR), and
exports them to Excel.

Usage
-----
.. code-block:: bash
    uv run extract-color-features  # demo dataset
    uv run extract-color-features \\
        --input ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/ \\
        --output ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/\
roi_10_color_features_with_ratio.xlsx
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd

from yheart._typing import AnnotationFile

# Default paths for Demo dataset
DEFAULT_INPUT_FOLDER = "./samples_images/results/Auto_segment_Yellow_heart/json/"
DEFAULT_OUTPUT_XLSX = "./samples_images/results/roi_10_color_features_with_ratio.xlsx"


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_YELLOW = "Yellow_area"
LABEL_PAN = "Pan_center_contour_area"


def round_and_clip_polygon(points: Sequence[Sequence[float]], w: int, h: int):
    """Round polygon vertices to integer pixels and clamp to the canvas."""
    pts: list[tuple[int, int]] = []
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        x, y = p[0], p[1]
        xi = max(0, min(w - 1, round(x)))
        yi = max(0, min(h - 1, round(y)))
        pts.append((xi, yi))
    if not pts:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def circular_mean_degrees(h_vals_0_179: np.ndarray):
    """Circular mean of OpenCV hue samples, in degrees."""
    if len(h_vals_0_179) == 0:
        return float("nan")

    # OpenCV HSV Hue is in [0, 179] (half-degree units); map to [0, 359].
    deg = np.asarray(h_vals_0_179, dtype=np.float64) * 2.0
    rad = np.deg2rad(deg)
    sinm = np.nanmean(np.sin(rad))
    cosm = np.nanmean(np.cos(rad))
    if np.isnan(sinm) or np.isnan(cosm):
        return float("nan")
    mean_rad = math.atan2(sinm, cosm)
    mean_deg = math.degrees(mean_rad)
    if mean_deg < 0:
        mean_deg += 360.0
    return mean_deg


def mean_safe(arr: np.ndarray):
    """NaN-tolerant mean (NaN when empty or all-NaN)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    a = a[np.isfinite(a)]
    return np.mean(a) if a.size > 0 else float("nan")


def compute_ExR_mean(R: np.ndarray, G: np.ndarray, B: np.ndarray):
    """Mean Excess Red Index: ExR = 1.4*r - g with r=R/S, g=G/S, S=R+G+B.

    Equivalent to (1.4*R - G) / S and consistent with the per-pixel ExR used
    in segmentation. A vanishing S is guarded by setting it to 1.0.
    """
    R_f, G_f, B_f = R.astype(np.float64), G.astype(np.float64), B.astype(np.float64)
    sum_RGB = R_f + G_f + B_f
    sum_RGB[sum_RGB == 0] = 1.0
    r, g = R_f / sum_RGB, G_f / sum_RGB
    ExR = 1.4 * r - g
    return mean_safe(ExR)


def process_image_pair(
    image_path: str | Path, json_path: str | Path
) -> dict[str, str | int | float]:
    """Extract the area ratio and ten color traits for one image/JSON pair."""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h, w = img.shape[:2]
    jd: AnnotationFile = json.loads(Path(json_path).read_text(encoding="utf-8"))
    mask_yellow = np.zeros((h, w), dtype=np.uint8)
    mask_pan = np.zeros((h, w), dtype=np.uint8)
    for shape in jd.get("shapes", []):
        label = shape.get("label")
        pts = shape.get("points", [])
        poly = round_and_clip_polygon(pts, w, h)
        if poly.shape[0] < 3:
            continue
        if label == LABEL_YELLOW:
            cv2.fillPoly(mask_yellow, [poly.reshape((-1, 1, 2))], 255)
        elif label == LABEL_PAN:
            cv2.fillPoly(mask_pan, [poly.reshape((-1, 1, 2))], 255)
    coords_yellow = np.where(mask_yellow == 255)
    yellow_pixel_count = len(coords_yellow[0])
    pan_pixel_count = int(np.sum(mask_pan == 255))
    yellow_ratio = yellow_pixel_count / pan_pixel_count if pan_pixel_count > 0 else 0.0
    if yellow_pixel_count == 0:
        raise ValueError(f"No yellow region pixels detected in {Path(image_path).name}")
    roi_pixels = img[coords_yellow]
    B_raw = roi_pixels[:, 0].astype(np.float64)
    G_raw = roi_pixels[:, 1].astype(np.float64)
    R_raw = roi_pixels[:, 2].astype(np.float64)
    R_mean, G_mean, B_mean = mean_safe(R_raw), mean_safe(G_raw), mean_safe(B_raw)
    ExR_mean = compute_ExR_mean(R_raw, G_raw, B_raw)
    hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[coords_yellow]
    lab_roi = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[coords_yellow]
    H_circular_mean_deg = circular_mean_degrees(hsv_roi[:, 0])
    S_mean = mean_safe(hsv_roi[:, 1].astype(np.float64) / 255.0)
    V_mean = mean_safe(hsv_roi[:, 2].astype(np.float64) / 255.0)
    L_mean = mean_safe(lab_roi[:, 0].astype(np.float64) / 255.0 * 100.0)
    a_mean = mean_safe(lab_roi[:, 1].astype(np.float64) - 128.0)
    b_mean = mean_safe(lab_roi[:, 2].astype(np.float64) - 128.0)
    return {
        "filename": Path(image_path).name,
        "Yellow_pixel_count": yellow_pixel_count,
        "Pan_pixel_count": pan_pixel_count,
        "Yellow_Ratio": yellow_ratio,
        "R_mean": R_mean,
        "G_mean": G_mean,
        "B_mean": B_mean,
        "H_circular_mean_deg": H_circular_mean_deg,
        "S_mean": S_mean,
        "V_mean": V_mean,
        "L_mean": L_mean,
        "a_mean": a_mean,
        "b_mean": b_mean,
        "ExR_mean": ExR_mean,
    }


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    input_folder, output_xlsx = args.input, args.output
    input_p = Path(input_folder)
    rows: list[dict[str, str | int | float]] = []
    files = sorted([p for p in input_p.iterdir() if p.suffix.lower() in VALID_EXTS])
    for img_path in files:
        json_path = input_p / f"{img_path.stem}.json"
        if not json_path.exists():
            continue
        try:
            rec = process_image_pair(str(img_path), str(json_path))
            rows.append(rec)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    if rows:
        save_results_to_excel(rows, output_xlsx)


def save_results_to_excel(rows: list[dict[str, str | int | float]], output_xlsx: str) -> None:
    df = pd.DataFrame(rows)
    priority_cols = ["filename", "Yellow_pixel_count", "Pan_pixel_count", "Yellow_Ratio"]
    other_cols = [c for c in df.columns if c not in priority_cols]
    Path(output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    df[priority_cols + other_cols].to_excel(output_xlsx, index=False)
    print(f"Feature extraction complete. Output saved to: {output_xlsx}")


@dataclass(frozen=True, slots=True)
class ExtractColorFeaturesArgs:
    """Parsed command-line arguments."""

    input: str
    output: str


def parse_args(argv: Sequence[str] | None = None) -> ExtractColorFeaturesArgs:
    p = argparse.ArgumentParser(description="Extract yellow-heart phenotypic color-space features.")
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT_FOLDER,
        help="Folder of images + Yellow_area/Pan JSON annotations.",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_XLSX, help="Output .xlsx path for color features."
    )
    return cast(ExtractColorFeaturesArgs, p.parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
