"""Reproduce the polygon rule in the archived 2026-11-16 `1_YellowHeart.py`."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from yheart._typing import AnnotationFile, ShapeRecord
from yheart.imaging.io import read_image, write_image
from yheart.imaging.segment_yellow_heart import (
    EXR_THRESHOLD,
    TARGET_LABEL,
    VALID_EXTS,
    YELLOW_LABEL,
    compute_exr_per_pixel,
    round_and_clip_polygon,
)

ENCODED_B_LOW = 20
ENCODED_B_HIGH = 70


def segment(img: np.ndarray, shapes: list[ShapeRecord]) -> np.ndarray | None:
    """Find the largest external contour after opening the archived threshold mask."""
    height, width = img.shape[:2]
    head = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        if shape.get("label") == TARGET_LABEL and shape.get("shape_type") == "polygon":
            polygon = round_and_clip_polygon(shape.get("points", []), width, height)
            if len(polygon) >= 3:
                cv2.fillPoly(head, [polygon], 255)
    if not head.any():
        return None

    exr = compute_exr_per_pixel(img)
    # The archived code compares OpenCV's encoded uint8 b channel directly.
    # Its range [20, 70] corresponds to CIELAB b* in [-108, -58].
    encoded_b = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2]
    selected = (exr > EXR_THRESHOLD) | (
        (encoded_b >= ENCODED_B_LOW) & (encoded_b <= ENCODED_B_HIGH)
    )
    mask = ((head == 255) & selected).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def run(input_dir: Path, output_dir: Path, annotation_dir: Path) -> None:
    """Write contour overlays and polygons in separate directories."""
    if len({directory.resolve() for directory in (input_dir, output_dir, annotation_dir)}) != 3:
        raise ValueError("Input, overlay and annotation directories must be distinct.")
    images = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for path in images:
        source = path.with_suffix(".json")
        if not source.exists():
            print(f"Skipping {path.name}: missing annotation.")
            continue
        image = read_image(path)
        if image is None:
            print(f"Skipping {path.name}: unreadable image.")
            continue
        annotation: AnnotationFile = json.loads(source.read_text(encoding="utf-8"))
        shapes = annotation.get("shapes", [])
        contour = segment(image, shapes)
        annotation["shapes"] = [shape for shape in shapes if shape.get("label") == TARGET_LABEL]
        annotation["imagePath"] = path.name
        annotation["imageHeight"], annotation["imageWidth"] = image.shape[:2]
        if contour is not None and len(contour) > 0:
            annotation["shapes"].append({
                "label": YELLOW_LABEL,
                "points": [[float(x), float(y)] for x, y in contour.reshape(-1, 2)],
                "group_id": None,
                "description": None,
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            })
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
        # These outputs are polygon-only, even when input has newer raster metadata.
        annotation.pop("region_masks", None)
        annotation.pop("yellow_mask_path", None)
        annotation.pop("valid_head_mask_path", None)
        annotation.pop("segmentation", None)
        destination = annotation_dir / source.name
        destination.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_image(output_dir / path.name, image)
        processed += 1
    if not processed:
        raise ValueError("No readable image and annotation pairs were found.")
    print(f"Processed {processed} image and annotation pairs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Images and head polygon JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Contour-overlay directory.")
    parser.add_argument("--json-output", type=Path, required=True, help="Output annotation folder.")
    args = parser.parse_args()
    run(args.input, args.output, args.json_output)


if __name__ == "__main__":
    main()
