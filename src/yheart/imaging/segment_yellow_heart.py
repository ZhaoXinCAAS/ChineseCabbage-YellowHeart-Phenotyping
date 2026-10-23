"""Automated extraction of the internal yellow-heart region.

Given cropped single-cabbage cross-section images plus their head-region
polygon JSON (label ``Pan_center_contour_area``), this script segments the
internal yellow-heart tissue using ExR, excluding the annotated short-stem
area, and exports both the
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
import csv
import json
import shutil
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
STEM_LABEL = "Short_stem"
ALGORITHM_VERSION = "2.1-exr-envelope"
EXR_THRESHOLD = 0.15
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
    # uint8 OpenCV Lab encodes b* as b* + 128. Cast before subtraction.
    return b_channel.astype(np.float32) - 128.0


def load_json_and_get_mask(json_path: str | Path, img_shape: tuple[int, ...]):
    """Rasterize the head-region polygon of an annotation JSON to a mask."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    stem_mask = np.zeros_like(mask)
    data: AnnotationFile = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if (data.get("imageHeight"), data.get("imageWidth")) != (h, w):
        raise ValueError(f"Annotation/image dimensions differ: {json_path}")
    for shape in data.get("shapes", []):
        if (
            shape.get("label") in (TARGET_LABEL, STEM_LABEL)
            and shape.get("shape_type") == "polygon"
        ):
            points = shape.get("points", [])
            poly = round_and_clip_polygon(points, w, h)
            if poly.shape[0] >= 3:
                target = mask if shape.get("label") == TARGET_LABEL else stem_mask
                cv2.fillPoly(target, [poly.reshape((-1, 1, 2))], 255)
    mask[stem_mask > 0] = 0
    return mask


def extract_yellow_contour(
    img: np.ndarray,
    mask: np.ndarray,
    exr_threshold: float = EXR_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Use ExR to locate the core; retain the largest external contour."""
    if not np.isfinite(exr_threshold):
        raise ValueError("ExR threshold must be finite")
    exr = compute_exr_per_pixel(img)
    masked_exr = np.where(mask == 255, exr, 0)
    yellow_binary = (masked_exr > exr_threshold).astype(np.uint8) * 255
    yellow_binary[mask != 255] = 0
    kernel = np.ones((3, 3), np.uint8)
    yellow_binary = cv2.erode(yellow_binary, kernel, iterations=1)
    yellow_binary = cv2.dilate(yellow_binary, kernel, iterations=1)
    yellow_binary[mask != 255] = 0
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
        shape
        for shape in json_data.get("shapes", [])
        if shape.get("label") in (TARGET_LABEL, STEM_LABEL)
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
    remove_gaps: bool = False,
) -> "dict[str, object]":
    """Segment one image/JSON pair end to end."""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"filename": Path(img_path).name, "status": "unreadable_image"}
    mask = load_json_and_get_mask(json_path, img.shape)
    contour, pixels = extract_yellow_contour(img, mask, exr_threshold)
    envelope = np.zeros_like(mask)
    if contour is not None:
        cv2.fillPoly(envelope, [contour], 255)
    envelope[mask == 0] = 0
    status = "ok" if envelope.any() else "no_yellow_detected"
    if not mask.any():
        status = "no_valid_head"
    # Gap removal is an optional sensitivity analysis, not a new default phenotype.
    regions = {"envelope": envelope, "pixels": pixels, "valid_head": mask}
    if remove_gaps:
        lightness = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 0]
        radius = max(2, round(8 * max(img.shape[:2]) / 1024))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        deficit = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, kernel)
        gaps = (envelope > 0) & (lightness < 50) & (deficit > 15)
        tissue = envelope.copy()
        tissue[gaps] = 0
        regions["tissue"] = tissue
    out = Path(output_json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, region in regions.items():
        relative = Path("masks") / f"{out.stem}_{name}.png"
        (out.parent / relative).parent.mkdir(parents=True, exist_ok=True)
        if not imwrite_unicode(out.parent / relative, region):
            raise OSError(f"Failed to write mask: {relative}")
        paths[name] = relative.as_posix()
    annotation = contour_to_json(contour, json_path, img_path, img.shape)
    annotation["region_masks"] = paths
    annotation["yellow_mask_path"] = paths["envelope"]
    annotation["valid_head_mask_path"] = paths["valid_head"]
    annotation["segmentation"] = {
        "algorithm_version": ALGORITHM_VERSION,
        "color_rule": "exr",
        "status": status,
        "exr_threshold": exr_threshold,
        "region_definition": "largest external contour, filled, clipped to head minus short stem",
        "gap_removal": remove_gaps,
        "gap_max_L": 50,
        "gap_min_blackhat": 15,
        "gap_radius_at_1024": 8,
        "opencv_version": cv2.__version__,
    }
    out.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    image_copy = out.parent / Path(img_path).name
    if image_copy.resolve() != Path(img_path).resolve():
        shutil.copyfile(img_path, image_copy)
    visualize_contour_and_save(img.copy(), contour, output_img_path)
    return {
        "filename": Path(img_path).name,
        "status": status,
        "valid_head_px": int((mask > 0).sum()),
        "yellow_px": int((envelope > 0).sum()),
    }


def batch_process(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    json_output_dir: str | Path = DEFAULT_JSON_OUTPUT_DIR,
    exr_threshold: float = EXR_THRESHOLD,
    remove_gaps: bool = False,
):
    """Segment every annotated image in a folder."""
    input_path = Path(input_dir)
    destination = Path(json_output_dir)
    if list(destination.glob("*.json")) or (destination / "segmentation_status.csv").exists():
        raise ValueError("Use a fresh JSON output directory to avoid mixing algorithm versions")
    img_files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTS)
    records: "list[dict[str, object]]" = []
    for img_file in img_files:
        json_file = input_path / f"{img_file.stem}.json"
        if not json_file.exists():
            records.append({"filename": img_file.name, "status": "missing_annotation"})
            continue
        output_img_file = Path(output_dir) / img_file.name
        output_json_file = Path(json_output_dir) / f"{img_file.stem}.json"
        try:
            records.append(
                process_single_file(
                    img_file,
                    json_file,
                    output_img_file,
                    output_json_file,
                    exr_threshold,
                    remove_gaps,
                )
            )
        except (ValueError, OSError, KeyError, cv2.error) as exc:
            records.append({
                "filename": img_file.name,
                "status": "processing_error",
                "error": str(exc),
            })
    report = Path(json_output_dir) / "segmentation_status.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "status",
                "valid_head_px",
                "yellow_px",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return records


@dataclass(frozen=True, slots=True)
class SegmentYellowHeartArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    json_output: str
    exr_threshold: float
    remove_gaps: bool


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
        "--remove-gaps",
        action="store_true",
        help="Also save exploratory tissue mask excluding dark-gap candidates.",
    )
    return cast(SegmentYellowHeartArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.json_output).mkdir(parents=True, exist_ok=True)
    records = batch_process(
        args.input,
        args.output,
        args.json_output,
        args.exr_threshold,
        args.remove_gaps,
    )
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"Segmentation finished: {len(records)} input images; statuses={counts}")


if __name__ == "__main__":
    main(sys.argv[1:])
