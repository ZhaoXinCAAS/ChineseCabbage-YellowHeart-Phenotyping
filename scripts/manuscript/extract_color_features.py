"""Reproduce the 2026-11-16 `2_Yellow_heart_color_extract_mim_max.py` study workbook."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from yheart._typing import AnnotationFile
from yheart.imaging.extract_color_features import (
    COLOR_FEATURE_NAMES,
    LABEL_PAN,
    LABEL_YELLOW,
    VALID_EXTS,
    color_features_from_mask,
    round_and_clip_polygon,
)
from yheart.imaging.io import read_image


def measure(image_path: Path, annotation_path: Path) -> dict[str, str | int | float]:
    """Measure the two archived polygons and extract color in the yellow polygon."""
    image = read_image(image_path)
    if image is None:
        raise ValueError("unreadable image")
    height, width = image.shape[:2]
    annotation: AnnotationFile = json.loads(annotation_path.read_text(encoding="utf-8"))
    masks = {
        label: np.zeros((height, width), dtype=np.uint8) for label in (LABEL_PAN, LABEL_YELLOW)
    }
    for shape in annotation.get("shapes", []):
        label = shape.get("label", "")
        if label in masks:
            polygon = round_and_clip_polygon(shape.get("points", []), width, height)
            if len(polygon) >= 3:
                cv2.fillPoly(masks[label], [polygon], 255)

    yellow_pixels = int(np.count_nonzero(masks[LABEL_YELLOW]))
    if not yellow_pixels:
        raise ValueError("no yellow polygon pixels")
    head_pixels = int(np.count_nonzero(masks[LABEL_PAN]))
    return {
        "filename": image_path.name,
        "Yellow_pixel_count": yellow_pixels,
        "Pan_pixel_count": head_pixels,
        "Yellow_Ratio": yellow_pixels / head_pixels if head_pixels else 0.0,
        **color_features_from_mask(image, masks[LABEL_YELLOW]),
    }


def run(annotation_dir: Path, image_dir: Path, output: Path) -> None:
    """Export raw and cohort-wide Min-Max traits in interleaved columns."""
    if not annotation_dir.is_dir():
        raise ValueError(f"Annotation directory does not exist: {annotation_dir}")
    rows: list[dict[str, str | int | float]] = []
    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS
    )
    for path in images:
        annotation = annotation_dir / path.with_suffix(".json").name
        if not annotation.is_file():
            print(f"Skipping {path.name}: missing annotation.")
            continue
        try:
            rows.append(measure(path, annotation))
        except (ValueError, OSError, cv2.error) as exc:
            print(f"Skipping {path.name}: {exc}")
    if not rows:
        raise ValueError("No images with measurable yellow polygons were found.")

    frame = pd.DataFrame(rows)
    features = list(COLOR_FEATURE_NAMES)
    scaled = MinMaxScaler().fit_transform(frame[features])
    columns = ["filename", "Yellow_pixel_count", "Pan_pixel_count", "Yellow_Ratio"]
    for index, feature in enumerate(features):
        normalized = f"{feature}_norm"
        frame[normalized] = scaled[:, index]
        columns.extend((feature, normalized))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_excel(output, index=False)
    print(f"Saved measurements for {len(rows)} images to {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Polygon JSON directory.")
    parser.add_argument("--images", type=Path, help="Original images. Defaults to --input.")
    parser.add_argument("--output", type=Path, required=True, help="Output .xlsx workbook.")
    args = parser.parse_args()
    run(args.input, args.images or args.input, args.output)


if __name__ == "__main__":
    main()
