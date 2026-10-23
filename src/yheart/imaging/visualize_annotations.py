r"""Render annotation polygons as semi-transparent review overlays.

Draws every polygon from an X-AnyLabeling JSON file on top of its image:
filled colour, closed outline, and a label tag. Useful for inspecting
predicted head / yellow-heart regions before or after segmentation, and for
building figure panels from a folder of image + JSON pairs.

Usage
-----
.. code-block:: bash
    uv run visualize-annotations
    uv run visualize-annotations --input ./data/annotations \
        --output ./data/annotations_vis --labels Yellow_area,Short_stem
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from yheart._typing import AnnotationFile
from yheart.imaging.extract_color_features import VALID_EXTS, round_and_clip_polygon
from yheart.imaging.io import read_image, write_image

DEFAULT_INPUT = "./samples_images/results/Auto_segment_Yellow_heart/json"
DEFAULT_OUTPUT = "./samples_images/results/Annotation_visualization"
DEFAULT_ALPHA = 0.35

# BGR colors, assigned to labels in first-seen order.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 105, 180),
    (0, 215, 255),
    (50, 205, 50),
    (238, 130, 238),
    (0, 165, 255),
)

Color = tuple[int, int, int]


def draw_annotations(
    image: np.ndarray,
    data: AnnotationFile,
    alpha: float = DEFAULT_ALPHA,
    palette: Sequence[Color] = PALETTE,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with polygon fills, outlines, and tags."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    height, width = image.shape[:2]
    json_width = int(data.get("imageWidth", 0) or 0)
    json_height = int(data.get("imageHeight", 0) or 0)
    canvas_w = min(width, json_width) if json_width > 0 else width
    canvas_h = min(height, json_height) if json_height > 0 else height

    out = image.copy()
    overlay = out.copy()
    wanted = set(labels) if labels else None
    label_colors: dict[str, Color] = {}

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "unknown"))
        if wanted is not None and label not in wanted:
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        points = shape.get("points")
        if not points or len(points) < 3:
            continue
        poly = round_and_clip_polygon(points, canvas_w, canvas_h)
        if poly.shape[0] < 3:
            continue
        pts = poly.reshape((-1, 1, 2))
        color = label_colors.setdefault(
            label, palette[len(label_colors) % len(palette)]
        )

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        origin = (int(poly[0][0]), int(poly[0][1]))
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            out,
            (origin[0], origin[1] - text_h - 4),
            (origin[0] + text_w + 4, origin[1] + baseline),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (origin[0] + 2, origin[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def visualize_folder(
    input_dir: str,
    output_dir: str,
    alpha: float = DEFAULT_ALPHA,
    labels: Sequence[str] | None = None,
    recursive: bool = False,
) -> tuple[list[Path], list[str]]:
    """Render every image with a JSON sidecar; return (written, skipped)."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    out_root = Path(output_dir)
    pattern = "**/*" if recursive else "*"
    images = sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    written: list[Path] = []
    skipped: list[str] = []
    for image_path in images:
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            skipped.append(f"{image_path.name}: missing JSON sidecar")
            continue
        image = read_image(image_path)
        if image is None:
            skipped.append(f"{image_path.name}: unreadable image")
            continue
        data = cast(AnnotationFile, json.loads(json_path.read_text(encoding="utf-8")))
        rendered = draw_annotations(image, data, alpha, labels=labels)
        relative = image_path.parent.relative_to(root) if recursive else Path()
        target = out_root / relative / f"{image_path.stem}_vis{image_path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_image(target, rendered)
        written.append(target)
    return written, skipped


@dataclass(frozen=True, slots=True)
class VisualizeAnnotationsArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    alpha: float
    labels: str | None
    recursive: bool


def parse_args(argv: Sequence[str] | None = None) -> VisualizeAnnotationsArgs:
    p = argparse.ArgumentParser(description="Render annotation polygons as PNG overlays.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Folder with images and JSON sidecars.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Folder for rendered overlays.")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Polygon fill opacity.")
    p.add_argument("--labels", default=None, help="Comma-separated labels to draw (default: all).")
    p.add_argument("--recursive", action="store_true", help="Walk subfolders.")
    return cast(VisualizeAnnotationsArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels else None
    try:
        written, skipped = visualize_folder(
            args.input, args.output, args.alpha, labels, args.recursive
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Rendered {len(written)} overlay(s) into {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for message in skipped[:10]:
            print(f"  - {message}")
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
