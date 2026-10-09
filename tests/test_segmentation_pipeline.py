"""Regression checks for actual phenotype masks and failure accounting."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yheart.imaging.segment_yellow_heart import (
    batch_process,
    compute_lab_b_channel,
    extract_yellow_contour,
    process_single_file,
)
from yheart.imaging.extract_color_features import process_image_pair
from yheart.imaging.evaluate_segmentation import mask_from_json


def make_input(root: Path, yellow: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    image = np.full((60, 60, 3), (20, 220, 240) if yellow else (240, 240, 240), np.uint8)
    image_path, json_path = root / "sample.png", root / "sample.json"
    cv2.imwrite(str(image_path), image)
    json_path.write_text(
        json.dumps({
            "imageHeight": 60,
            "imageWidth": 60,
            "shapes": [
                {
                    "label": "Pan_center_contour_area",
                    "shape_type": "polygon",
                    "points": [[5, 5], [54, 5], [54, 54], [5, 54]],
                },
                {
                    "label": "Short_stem",
                    "shape_type": "polygon",
                    "points": [[25, 25], [34, 25], [34, 34], [25, 34]],
                },
            ],
        }),
        encoding="utf-8",
    )
    return image_path, json_path


def test_lab_star_and_roi_are_correct():
    image = np.array([[[128, 128, 128], [30, 220, 240]]], dtype=np.uint8)
    b = compute_lab_b_channel(image)
    assert abs(b[0, 0]) <= 1 and b[0, 1] > 20
    # Even a negative threshold must not select pixels outside the ROI.
    _, mask = extract_yellow_contour(
        np.tile(image, (5, 3, 1)), np.zeros((5, 6), np.uint8), exr_threshold=-1
    )
    assert not mask.any()


def test_default_does_not_expand_to_lab_only_peripheral_leaves():
    image = np.full((60, 60, 3), (40, 180, 160), np.uint8)
    image[20:40, 20:40] = (30, 200, 220)
    roi = np.full((60, 60), 255, np.uint8)
    _, default = extract_yellow_contour(image, roi)
    assert default[30, 30] == 255
    assert default[10, 10] == 0
    assert (default > 0).sum() == 400
    with pytest.raises(ValueError, match="must be finite"):
        extract_yellow_contour(image, roi, exr_threshold=float("nan"))


def test_export_features_and_evaluation_share_exact_mask(tmp_path: Path):
    image, annotation = make_input(tmp_path / "input")
    original = image.read_bytes()
    output = tmp_path / "out" / "sample.json"
    result = process_single_file(image, annotation, tmp_path / "view.png", output, remove_gaps=True)
    assert result["status"] == "ok"
    mask = mask_from_json(output)
    assert mask.sum() == 2400  # 50*50 head minus 10*10 stem
    assert not mask[25:35, 25:35].any()
    features = process_image_pair(output.with_suffix(".png"), output)
    assert features["Yellow_pixel_count"] == mask.sum()
    assert features["Pan_pixel_count"] == 2400
    assert features["Yellow_Ratio"] == 1
    assert image.read_bytes() == original
    data = json.loads(output.read_text())
    assert "tissue" in data["region_masks"]
    assert any(s["label"] == "Short_stem" for s in data["shapes"])


def test_empty_yellow_and_missing_annotation_are_retained(tmp_path: Path):
    image, _ = make_input(tmp_path / "input", yellow=False)
    cv2.imwrite(str(image.parent / "unannotated.png"), np.zeros((5, 5, 3), np.uint8))
    out = tmp_path / "out"
    records = batch_process(image.parent, tmp_path / "visual", out)
    assert {r["status"] for r in records} == {"no_yellow_detected", "missing_annotation"}
    features = process_image_pair(out / "sample.png", out / "sample.json")
    assert features["Yellow_Ratio"] == 0
    assert np.isnan(features["b_mean"])
    assert (out / "segmentation_status.csv").exists()
    with pytest.raises(ValueError, match="fresh JSON"):
        batch_process(image.parent, tmp_path / "visual", out)
