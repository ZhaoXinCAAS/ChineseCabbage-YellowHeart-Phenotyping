"""Tests for annotation overlay rendering."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


from yheart._typing import AnnotationFile, ShapeRecord
from yheart.imaging.io import read_image
from yheart.imaging.visualize_annotations import (
    draw_annotations,
    visualize_folder,
)


def _pair(folder: Path, stem: str, shapes: list[ShapeRecord]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    image_path = folder / f"{stem}.png"
    cv2.imwrite(str(image_path), image)
    (folder / f"{stem}.json").write_text(
        json.dumps({"imageHeight": 40, "imageWidth": 40, "shapes": shapes}),
        encoding="utf-8",
    )
    return image_path


def _polygon(label: str, points: list[list[float]]) -> ShapeRecord:
    return {"label": label, "shape_type": "polygon", "points": points}


def test_draw_polygon_fills_and_outlines():
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    data: AnnotationFile = {
        "imageHeight": 40,
        "imageWidth": 40,
        "shapes": [_polygon("Yellow_area", [[5, 5], [35, 5], [35, 35], [5, 35]])],
    }
    out = draw_annotations(image, data, alpha=0.35)
    assert out.shape == image.shape
    assert not np.array_equal(out[20, 20], image[20, 20])
    # The source image is never modified in place.
    assert np.array_equal(image, np.full((40, 40, 3), 255, dtype=np.uint8))


def test_label_filter_skips_unrequested_shapes():
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    data: AnnotationFile = {
        "imageHeight": 40,
        "imageWidth": 40,
        "shapes": [
            _polygon("Yellow_area", [[2, 2], [18, 2], [18, 18], [2, 18]]),
            _polygon("Short_stem", [[22, 22], [38, 22], [38, 38], [22, 38]]),
        ],
    }
    out = draw_annotations(image, data, labels=["Short_stem"])
    assert np.array_equal(out[10, 10], image[10, 10])
    assert not np.array_equal(out[30, 30], image[30, 30])


def test_invalid_alpha_rejected():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    data: AnnotationFile = {"imageHeight": 10, "imageWidth": 10, "shapes": []}
    with pytest.raises(ValueError, match="alpha"):
        draw_annotations(image, data, alpha=1.5)


def test_visualize_folder_writes_outputs_and_skips_without_json(tmp_path: Path):
    source = tmp_path / "annotations"
    _pair(source, "a", [_polygon("Yellow_area", [[5, 5], [35, 5], [35, 35], [5, 35]])])
    cv2.imwrite(str(source / "no_json.png"), np.zeros((20, 20, 3), dtype=np.uint8))

    written, skipped = visualize_folder(str(source), str(tmp_path / "vis"))
    assert [p.name for p in written] == ["a_vis.png"]
    assert written[0].exists()
    assert any("JSON" in message for message in skipped)

    rendered = read_image(written[0])
    assert rendered is not None
    assert rendered.shape == (40, 40, 3)
    assert not np.array_equal(rendered[20, 20], np.full(3, 255, dtype=np.uint8))


def test_recursive_mode_mirrors_subfolders(tmp_path: Path):
    source = tmp_path / "annotations"
    _pair(source / "Grade_2", "a", [_polygon("Yellow_area", [[5, 5], [35, 5], [35, 35], [5, 35]])])
    written, _ = visualize_folder(str(source), str(tmp_path / "vis"), recursive=True)
    assert written == [tmp_path / "vis" / "Grade_2" / "a_vis.png"]
