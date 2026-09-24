r"""Crop individual Chinese cabbage head ROIs from raw cross-section images.

Given a folder of X-AnyLabeling/SAM 3 polygon JSON annotations and their
corresponding images, this script crops each annotated leaf-head region into a
separate image, expanding the bounding box by a configurable margin.

Usage
-----
.. code-block:: bash
    uv run crop-heads  # demo dataset (defaults)
    uv run crop-heads --input ./data/SAM3_predicted \
        --images ./data/SAM3_predicted \
        --output ./data/CROP_black  # Figshare full dataset
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
from numpy import int32, zeros

from yheart._typing import AnnotationFile, ShapeRecord

# Default paths for Demo dataset
DEFAULT_JSON_FOLDER = "./samples_images/results/SAM3_predicted"
DEFAULT_IMAGE_FOLDER = "./samples_images/results/SAM3_predicted"
DEFAULT_OUTPUT_DIR = "./samples_images/results/annotations_head_region"
DEFAULT_EXPAND_RATIO = 0.2


def crop_single_image(
    img: np.ndarray,
    shapes: Sequence[ShapeRecord],
    img_base_name: str,
    target_image_filename: str,
    output_root_dir: str,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
):
    """Crop every polygon ROI from *img* and write cropped images to disk."""
    object_count = 1
    shape_count = len(shapes)
    saved_paths: list[str] = []
    output_root = Path(output_root_dir)
    for region in shapes:
        coords = region.get("points", [])
        if len(coords) < 3:
            continue
        pts = np.array(coords, int32).reshape((-1, 1, 2))
        mask = zeros(img.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [pts], 255)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        x, y, w, h = cv2.boundingRect(pts)
        dx = int(w * expand_ratio)
        dy = int(h * expand_ratio)
        x = max(x - dx, 0)
        y = max(y - dy, 0)
        w = min(w + 2 * dx, img.shape[1] - x)
        h = min(h + 2 * dy, img.shape[0] - y)
        roi = masked_img[y : y + h, x : x + w]
        if roi.size == 0:
            print(f"Warning: Cropped region {object_count} in {target_image_filename} is empty.")
            object_count += 1
            continue
        if shape_count == 1:
            output_filename = target_image_filename
        else:
            file_ext = Path(target_image_filename).suffix
            output_filename = f"{img_base_name}_{object_count}{file_ext}"
        output_path = output_root / output_filename
        if cv2.imwrite(str(output_path), roi):
            print(f"Saved: {output_path}")
            saved_paths.append(str(output_path))
        else:
            print(f"Failed to save: {output_path}")
        object_count += 1
    return saved_paths


def crop_batch(
    json_folder_path: str,
    image_folder_path: str,
    output_root_dir: str,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
):
    """Crop head ROIs for every annotated image in a folder."""
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)
    for json_path in sorted(Path(json_folder_path).glob("*.json")):
        try:
            data: AnnotationFile = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"Error reading {json_path.name}: {e}")
            continue
        img_base_name = json_path.stem
        match = next(
            (
                p
                for p in Path(image_folder_path).iterdir()
                if p.is_file() and p.stem == img_base_name
            ),
            None,
        )
        if match is None:
            print(f"Image not found for {img_base_name}")
            continue
        img = cv2.imread(str(match))
        if img is None:
            print(f"Failed to load image: {match}")
            continue
        crop_single_image(
            img,
            data.get("shapes", []),
            img_base_name,
            match.name,
            output_root_dir,
            expand_ratio,
        )
    print("Processing complete.")


@dataclass(frozen=True, slots=True)
class CropHeadsArgs:
    """Parsed command-line arguments."""

    input: str
    images: str
    output: str
    expand_ratio: float


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(
        description="Crop individual Chinese cabbage head ROIs from "
        "JSON-annotated cross-section images."
    )
    p.add_argument(
        "--input", default=DEFAULT_JSON_FOLDER, help="Folder containing JSON annotations."
    )
    p.add_argument(
        "--images",
        default=DEFAULT_IMAGE_FOLDER,
        help="Folder containing the raw images (may equal --input).",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder for cropped head images."
    )
    p.add_argument(
        "--expand-ratio",
        type=float,
        default=DEFAULT_EXPAND_RATIO,
        help="Bounding-box margin expansion ratio (default: 0.2).",
    )
    return cast(CropHeadsArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    crop_batch(args.input, args.images, args.output, args.expand_ratio)


if __name__ == "__main__":
    main(sys.argv[1:])
