"""Tests for evaluate_segmentation (R1-1/Q8, R2-4)."""

import json
from pathlib import Path

import numpy as np
import pytest


from yheart.imaging.evaluate_segmentation import (
    auto_stratum,
    classify_cause,
    evaluate_dirs,
    iou_dice,
)


def _json(path: Path, label: str, points: list[list[int]], h: int = 20, w: int = 20):
    path.write_text(
        json.dumps({
            "imageHeight": h,
            "imageWidth": w,
            "shapes": [{"label": label, "shape_type": "polygon", "points": points}],
        }),
        encoding="utf-8",
    )


def test_iou_dice_perfect_and_empty():
    a = np.zeros((5, 5), bool)
    a[1:3, 1:3] = True
    iou, dice, _, _, _ = iou_dice(a, a.copy())
    assert iou == 1.0 and dice == 1.0
    e = np.zeros((5, 5), bool)
    iou, dice, _, _, _ = iou_dice(e, e)
    assert iou == 1.0 and dice == 1.0
    iou, dice, _, _, _ = iou_dice(a, e)
    assert iou == 0.0 and dice == 0.0


def test_classify_cause_requires_review_for_biology():
    assert classify_cause(0.0, 0, 5000) == "miss-no-pred:review-required"
    assert classify_cause(0.0, 0, 10) == "miss-no-pred:review-required"
    assert classify_cause(0.9, 100, 100) == "ok"
    assert "under-segment" in classify_cause(0.1, 100, 5000)
    assert "over-segment" in classify_cause(0.1, 5000, 2000)


def test_evaluate_dirs_end_to_end_and_strata(tmp_path: Path):
    pred_d, gt_d = tmp_path / "pred", tmp_path / "gt"
    pred_d.mkdir()
    gt_d.mkdir()
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    small = [[2, 2], [2, 4], [4, 4], [4, 2]]
    _json(pred_d / "a.json", "Yellow_area", box)
    _json(gt_d / "a.json", "Yellow_area", box)
    _json(pred_d / "b.json", "Yellow_area", [])  # miss
    _json(gt_d / "b.json", "Yellow_area", box)
    _json(pred_d / "c.json", "Yellow_area", small)
    _json(gt_d / "c.json", "Yellow_area", small)
    strata = tmp_path / "strata.csv"
    strata.write_text("filename,stratum\na.json,simple\nb.json,hard\nc.json,blurry\n")
    df = evaluate_dirs(str(pred_d), str(gt_d), strata_csv=str(strata))
    assert len(df) == 3
    assert float(df.loc[df["filename"] == "a.json", "iou"].iloc[0]) == 1.0
    assert bool(df.loc[df["filename"] == "b.json", "failure"].iloc[0])
    assert df.loc[df["filename"] == "b.json", "stratum"].iloc[0] == "hard"
    assert auto_stratum(10, 10, 0.9) == "unassigned"
    assert auto_stratum(5000, 100, 0.01) == "unassigned"


def test_missing_prediction_is_failure_even_for_empty_gt(tmp_path: Path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    _json(gt / "absent.json", "Yellow_area", [])
    _json(gt / "yellow.json", "Yellow_area", [[1, 1], [1, 5], [5, 5], [5, 1]])
    df = evaluate_dirs(str(pred), str(gt))
    assert len(df) == 2
    assert df["failure"].all() and df["prediction_missing"].all()
    assert (df["iou"] == 0).all()


def test_self_comparison_and_coordinate_drift_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="must be different"):
        evaluate_dirs(str(tmp_path), str(tmp_path))
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    _json(pred / "a.json", "Yellow_area", [], h=10)
    _json(gt / "a.json", "Yellow_area", [], h=20)
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_dirs(str(pred), str(gt))


def test_predictions_without_gt_are_not_silently_dropped(tmp_path: Path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    _json(pred / "extra.json", "Yellow_area", [])
    with pytest.raises(ValueError, match="without ground truth"):
        evaluate_dirs(str(pred), str(gt))
