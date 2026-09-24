"""Tests for color_calibrate + robustness_test (R1-6, R2-2)."""

import json
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


from yheart.imaging.color_calibrate import (
    apply_brightness,
    apply_ccm,
    apply_temp_shift,
    gray_world_wb,
)
from yheart.scoring.robustness_test import run_robustness


def _cast_image() -> npt.NDArray[np.uint8]:
    img = np.full((64, 64, 3), (60, 90, 40), dtype=np.uint8)  # dark green bg
    img[20:44, 20:44] = (30, 200, 220)  # yellow patch (BGR)
    return img


def _sidecar(path: Path, h: int = 64, w: int = 64):
    full = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    rect = [[20, 20], [20, 43], [43, 43], [43, 20]]
    path.write_text(
        json.dumps({
            "imageHeight": h,
            "imageWidth": w,
            "shapes": [
                {"label": "Pan_center_contour_area", "shape_type": "polygon", "points": full},
                {"label": "Yellow_area", "shape_type": "polygon", "points": rect},
            ],
        }),
        encoding="utf-8",
    )


def test_grayworld_equalizes_cast_means():
    img = _cast_image().astype(np.float64)
    img[:, :, 0] *= 1.6  # heavy blue cast
    img = np.clip(img, 0, 255).astype(np.uint8)
    before = img.reshape(-1, 3).mean(axis=0)
    assert before.max() - before.min() > 20
    out = gray_world_wb(img).reshape(-1, 3).mean(axis=0)
    # uint8 rounding leaves a small residual; the cast is still corrected.
    assert out.max() - out.min() < 6.0


def test_identity_perturbations_are_noops():
    img = _cast_image()
    assert np.array_equal(apply_brightness(img, 1.0), img)
    assert np.array_equal(apply_temp_shift(img, 1.0), img)
    assert np.array_equal(apply_ccm(img, np.eye(3)), img)


def test_temp_shift_moves_r_vs_b():
    img = _cast_image()
    warm = apply_temp_shift(img, 1.3)
    assert warm[:, :, 2].mean() >= img[:, :, 2].mean()
    assert warm[:, :, 0].mean() <= img[:, :, 0].mean()
    bright = apply_brightness(img, 1.3)
    hsv0 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    hsv1 = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    assert hsv1 > hsv0


def test_robustness_smoke_on_synthetic_pair(tmp_path: Path):
    for i in range(2):
        p = tmp_path / f"head_{i}.jpg"
        cv2.imwrite(str(p), _cast_image())
        _sidecar(tmp_path / f"head_{i}.json")
    summary, scaler_df = run_robustness(str(tmp_path), sample=2, seed=0)
    assert summary is not None and not summary.empty
    assert scaler_df is not None and not scaler_df.empty
    assert "wb-grayworld" in set(summary["perturb"])
    ident = summary.loc[summary["perturb"] == "identity"].iloc[0]
    assert ident["mean_seg_iou"] == 1.0
    assert ident["fail_rate"] == 0.0
    assert set(scaler_df["trait"]) == {"Yellow_Ratio", "S_mean", "b_mean", "ExR_mean", "B_mean"}
    assert set(scaler_df["scaler"]) == {"minmax", "robust"}
