"""Automated extraction of the internal yellow-heart region.

Given cropped single-cabbage cross-section images plus their head-region
polygon JSON (label ``Pan_center_contour_area``), this script segments the
internal yellow-heart tissue using the Excess Red Index (ExR) and CIELAB b*
thresholds, excluding the annotated short-stem area, and exports both the
visualised contour image and the yellow-area polygon JSON.

Usage
-----
.. code-block:: bash
    uv run segment-yellow-heart  # demo dataset
    uv run segment-yellow-heart \\
        --input ./data/annotations_head_region \\
        --output ./data/Auto_segment_Yellow_heart/True_Yellow_heart/Visualisation \\
        --json-output ./data/Auto_segment_Yellow_heart/True_Yellow_heart/\
json  # Figshare full dataset
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from yheart._typing import AnnotationFile, ShapeRecord

# Default paths for Demo dataset
DEFAULT_INPUT_DIR = "./samples_images/results/annotations_head_region"
DEFAULT_OUTPUT_DIR = "./samples_images/results/Auto_segment_Yellow_heart/"
DEFAULT_JSON_OUTPUT_DIR = "./samples_images/results/Auto_segment_Yellow_heart/json"


TARGET_LABEL = "Pan_center_contour_area"
EXR_THRESHOLD = 0.15
LAB_B_LOW_THRESHOLD = 20
LAB_B_HIGH_THRESHOLD = 70
EPS = 1e-6
YELLOW_LABEL = "Yellow_area"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def imwrite_unicode(path: str | Path, img: np.ndarray):
    """Write an image to a path that may contain non-ASCII characters.

    `cv2.imwrite` does not support non-ASCII paths on Windows; encoding via
    in-memory buffer avoids the limitation.
    """
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def round_and_clip_polygon(points: Sequence[Sequence[float]], w: int, h: int):
    """Round polygon vertices to integer pixels and clamp to the canvas."""
    pts: list[tuple[int, int]] = []
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        x, y = p[0], p[1]
        xi = round(x)
        yi = round(y)
        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))
        pts.append((xi, yi))
    if not pts:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def compute_exr_per_pixel(img: np.ndarray):
    """Excess Red Index per pixel: ExR = (1.4*R - G)/(R+G+B) with safe denom."""
    B = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    R = img[:, :, 2].astype(np.float64)
    S = R + G + B
    S_safe = np.where(S == 0, EPS, S)
    return (1.4 * R - G) / (S_safe + EPS)


def compute_lab_b_channel(img: np.ndarray):
    """CIELAB b* channel of a BGR image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    _, _, b_channel = cv2.split(lab)
    return b_channel


def load_json_and_get_mask(json_path: str | Path, img_shape: tuple[int, ...]):
    """Rasterize the head-region polygon of an annotation JSON to a mask."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    data: AnnotationFile = json.loads(Path(json_path).read_text(encoding="utf-8"))
    for shape in data.get("shapes", []):
        if shape.get("label") == TARGET_LABEL and shape.get("shape_type") == "polygon":
            points = shape.get("points", [])
            poly = round_and_clip_polygon(points, w, h)
            if poly.shape[0] >= 3:
                cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 255)
    return mask


def extract_yellow_contour(
    img: np.ndarray,
    mask: np.ndarray,
    exr_threshold: float = EXR_THRESHOLD,
    lab_b_low: float = LAB_B_LOW_THRESHOLD,
    lab_b_high: float = LAB_B_HIGH_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Segment yellow-heart tissue via ExR/b* thresholds; largest contour wins."""
    exr = compute_exr_per_pixel(img)
    lab_b = compute_lab_b_channel(img)
    masked_exr = np.where(mask == 255, exr, 0)
    masked_lab_b = np.where(mask == 255, lab_b, 0)
    exr_mask = (masked_exr > exr_threshold).astype(np.uint8) * 255
    lab_b_mask = np.where(
        (masked_lab_b >= lab_b_low) & (masked_lab_b <= lab_b_high), 255, 0
    ).astype(np.uint8)
    yellow_binary = cv2.bitwise_or(exr_mask, lab_b_mask)
    kernel = np.ones((3, 3), np.uint8)
    yellow_binary = cv2.erode(yellow_binary, kernel, iterations=1)
    yellow_binary = cv2.dilate(yellow_binary, kernel, iterations=1)
    contours, _ = cv2.findContours(yellow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea) if contours else None
    return max_contour, yellow_binary


def visualize_contour_and_save(
    img: np.ndarray, contour: np.ndarray | None, output_path: str | Path
):
    """Draw the contour on a copy and save the visualization image."""
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imwrite_unicode(output_path, img)


def contour_to_json(
    contour: np.ndarray | None,
    orig_json_path: str | Path,
    img_path: str | Path,
    img_shape: tuple[int, ...],
) -> AnnotationFile:
    """Rebuild the annotation JSON, preserving the head shape plus Yellow_area."""
    h, w = img_shape[:2]
    json_data: AnnotationFile = json.loads(Path(orig_json_path).read_text(encoding="utf-8"))
    preserved_shapes: list[ShapeRecord] = []
    preserved_shapes.extend(
        shape for shape in json_data.get("shapes", []) if shape.get("label") == TARGET_LABEL
    )
    json_data["shapes"] = preserved_shapes
    json_data["imagePath"] = Path(img_path).name
    json_data["imageHeight"] = h
    json_data["imageWidth"] = w
    if contour is not None and len(contour) > 0:
        points = [[float(point[0][0]), float(point[0][1])] for point in contour]
        yellow_shape: ShapeRecord = {
            "label": YELLOW_LABEL,
            "points": points,
            "group_id": None,
            "description": None,
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
        }
        json_data["shapes"].append(yellow_shape)
    return json_data


def save_yellow_area_json(
    contour: np.ndarray | None,
    orig_json_path: str | Path,
    img_path: str | Path,
    img_shape: tuple[int, ...],
    json_output_path: str | Path,
):
    """Write the Yellow_area polygon JSON to disk."""
    json_data = contour_to_json(contour, orig_json_path, img_path, img_shape)
    out_path = Path(json_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_single_file(
    img_path: str | Path,
    json_path: str | Path,
    output_img_path: str | Path,
    output_json_path: str | Path,
    exr_threshold: float = EXR_THRESHOLD,
    lab_b_low: float = LAB_B_LOW_THRESHOLD,
    lab_b_high: float = LAB_B_HIGH_THRESHOLD,
):
    """Segment one image/JSON pair end to end."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error loading image: {img_path}")
        return
    mask = load_json_and_get_mask(json_path, img.shape)
    if np.sum(mask) == 0:
        visualize_contour_and_save(img, None, output_img_path)
        save_yellow_area_json(None, json_path, img_path, img.shape, output_json_path)
        return
    contour, _ = extract_yellow_contour(img, mask, exr_threshold, lab_b_low, lab_b_high)
    visualize_contour_and_save(img, contour, output_img_path)
    save_yellow_area_json(contour, json_path, img_path, img.shape, output_json_path)


def batch_process(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    json_output_dir: str | Path = DEFAULT_JSON_OUTPUT_DIR,
    exr_threshold: float = EXR_THRESHOLD,
    lab_b_low: float = LAB_B_LOW_THRESHOLD,
    lab_b_high: float = LAB_B_HIGH_THRESHOLD,
):
    """Segment every annotated image in a folder."""
    input_path = Path(input_dir)
    img_files = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTS]
    for img_file in img_files:
        json_file = input_path / f"{img_file.stem}.json"
        if not json_file.exists():
            continue
        output_img_file = Path(output_dir) / img_file.name
        output_json_file = Path(json_output_dir) / f"{img_file.stem}.json"
        process_single_file(
            img_file,
            json_file,
            output_img_file,
            output_json_file,
            exr_threshold,
            lab_b_low,
            lab_b_high,
        )


@dataclass(frozen=True, slots=True)
class SegmentYellowHeartArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    json_output: str
    exr_threshold: float
    lab_b_low: float
    lab_b_high: float


def parse_args(argv: Sequence[str] | None = None) -> SegmentYellowHeartArgs:
    p = argparse.ArgumentParser(
        description="Automated yellow-heart region segmentation from "
        "annotated cross-section images."
    )
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT_DIR,
        help="Folder of images + head-region JSON annotations.",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder for visualised contour images."
    )
    p.add_argument(
        "--json-output",
        default=DEFAULT_JSON_OUTPUT_DIR,
        help="Output folder for the yellow-area polygon JSON.",
    )
    p.add_argument(
        "--exr-threshold",
        type=float,
        default=EXR_THRESHOLD,
        help="Excess Red Index threshold (default: 0.15).",
    )
    p.add_argument(
        "--lab-b-low",
        type=float,
        default=LAB_B_LOW_THRESHOLD,
        help="CIELAB b* low threshold (default: 20).",
    )
    p.add_argument(
        "--lab-b-high",
        type=float,
        default=LAB_B_HIGH_THRESHOLD,
        help="CIELAB b* high threshold (default: 70).",
    )
    return cast(SegmentYellowHeartArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.json_output).mkdir(parents=True, exist_ok=True)
    batch_process(
        args.input,
        args.output,
        args.json_output,
        args.exr_threshold,
        args.lab_b_low,
        args.lab_b_high,
    )
    print("Yellow-heart region segmentation complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
