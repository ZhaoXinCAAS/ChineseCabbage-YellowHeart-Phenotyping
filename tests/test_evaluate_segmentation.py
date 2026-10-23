"""Tests for evaluate_segmentation (R1-1/Q8, R2-4)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from yheart.imaging.evaluate_segmentation import (
    GRADE_LABELS,
    auto_stratum,
    classify_cause,
    evaluate_annotator_agreement,
    evaluate_dirs,
    iou_dice,
    main as evaluate_main,
    parse_grades,
    summarize_agreement,
    summarize_by_grade,
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


def test_gt_subdirs_adds_grade_and_aggregates(tmp_path: Path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    (gt / "Grade_2").mkdir(parents=True)
    (gt / "Grade_3").mkdir(parents=True)
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    small = [[2, 2], [2, 4], [4, 4], [4, 2]]
    _json(pred / "a.json", "Yellow_area", box)
    _json(gt / "Grade_2" / "a.json", "Yellow_area", box)
    _json(pred / "b.json", "Yellow_area", small)
    _json(gt / "Grade_3" / "b.json", "Yellow_area", box)

    df = evaluate_dirs(str(pred), str(gt), gt_subdirs=True)
    assert dict(zip(df["filename"], df["grade"])) == {
        "a.json": "Grade_2",
        "b.json": "Grade_3",
    }
    summary = summarize_by_grade(df)
    assert list(summary["grade"]) == ["Grade_2", "Grade_3", "ALL"]
    assert summary.loc[summary["grade"] == "Grade_2", "mean_iou"].iloc[0] == 1.0
    assert summary.loc[summary["grade"] == "Grade_3", "mean_iou"].iloc[0] < 0.5
    assert summary.loc[summary["grade"] == "ALL", "n"].iloc[0] == 2


def test_gt_subdirs_rejects_duplicate_stems(tmp_path: Path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    (gt / "Grade_2").mkdir(parents=True)
    (gt / "Grade_3").mkdir(parents=True)
    _json(pred / "a.json", "Yellow_area", [])
    _json(gt / "Grade_2" / "a.json", "Yellow_area", [])
    _json(gt / "Grade_3" / "a.json", "Yellow_area", [])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_dirs(str(pred), str(gt), gt_subdirs=True)


def test_cli_writes_grade_summary_and_excel(tmp_path: Path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir()
    (gt / "Grade_2").mkdir(parents=True)
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    _json(pred / "a.json", "Yellow_area", box)
    _json(gt / "Grade_2" / "a.json", "Yellow_area", box)
    summary_csv = tmp_path / "summary.csv"
    grade_csv = tmp_path / "grades.csv"
    excel = tmp_path / "report.xlsx"
    evaluate_main([
        "--pred-dir", str(pred),
        "--gt-dir", str(gt),
        "--gt-subdirs",
        "--output-csv", str(tmp_path / "eval.csv"),
        "--output-summary", str(summary_csv),
        "--output-grade-summary", str(grade_csv),
        "--excel", str(excel),
    ])
    assert grade_csv.exists() and excel.exists()
    grades = pd.read_csv(grade_csv)
    assert set(grades["grade"]) == {"Grade_2", "ALL"}
    sheets = pd.read_excel(excel, sheet_name=None)
    assert set(sheets) == {"Table_Summary", "Per_Sample_Details"}


def _graded_pred_bundle(root: Path) -> tuple[Path, Path]:
    """Prediction tree graded by folder + flat ground truth (revision layout)."""
    pred = root / "pred"
    gt = root / "gt"
    for grade in ("Grade_1", "Grade_2", "Grade_3"):
        (pred / grade).mkdir(parents=True)
    gt.mkdir()
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    small = [[2, 2], [2, 4], [4, 4], [4, 2]]
    _json(gt / "white.json", "Yellow_area", box)
    _json(gt / "light.json", "Yellow_area", box)
    _json(gt / "yellow.json", "Yellow_area", box)
    _json(pred / "Grade_1" / "white.json", "Yellow_area", box)
    _json(pred / "Grade_2" / "light.json", "Yellow_area", box)
    _json(pred / "Grade_3" / "yellow.json", "Yellow_area", small)
    return pred, gt


def test_grade_from_prediction_folders(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    df = evaluate_dirs(str(pred), str(gt), grade_from="pred")
    assert dict(zip(df["filename"], df["grade"])) == {
        "light.json": "Grade_2",
        "white.json": "Grade_1",
        "yellow.json": "Grade_3",
    }


def test_grade_from_pred_ignores_gt_samples_outside_the_benchmark(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    _json(gt / "unused.json", "Yellow_area", box)
    df = evaluate_dirs(str(pred), str(gt), grade_from="pred")
    assert "unused.json" not in set(df["filename"])
    assert len(df) == 3
    # Without an explicit grade source the whole GT folder is evaluated and the
    # unpaired sample counts as a missing prediction, keeping the completeness
    # guarantee for R1-1.
    full = evaluate_dirs(str(pred), str(gt))
    assert len(full) == 4
    unused = full.loc[full["filename"] == "unused.json"].iloc[0]
    assert bool(unused["prediction_missing"]) is True
    assert float(unused["iou"]) == 0.0


def test_grades_filter_keeps_only_requested(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    df = evaluate_dirs(
        str(pred), str(gt), grade_from="pred", grades=("Grade_2", "Grade_3")
    )
    assert set(df["grade"]) == {"Grade_2", "Grade_3"}
    summary = summarize_by_grade(df)
    assert list(summary["grade"]) == ["Grade_2", "Grade_3", "ALL"]
    assert float(summary.loc[summary["grade"] == "ALL", "mean_iou"].iloc[0]) < 0.8


def test_grades_filter_rejects_unknown_grade(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    with pytest.raises(ValueError, match="no samples"):
        evaluate_dirs(str(pred), str(gt), grade_from="pred", grades=("Grade_9",))


def test_grades_require_a_grade_source(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    with pytest.raises(ValueError, match="needs grade information"):
        evaluate_dirs(str(pred), str(gt), grades=("Grade_2",))
    with pytest.raises(ValueError, match="requires gt_subdirs"):
        evaluate_dirs(str(pred), str(gt), grade_from="gt")


def test_grade_labels_use_paper_names(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    df = evaluate_dirs(str(pred), str(gt), grade_from="pred", grades=("Grade_2",))
    summary = summarize_by_grade(df)
    label = summary.loc[summary["grade"] == "Grade_2", "grade_label"].iloc[0]
    assert label == GRADE_LABELS["Grade_2"] == "Grade 2 (Light Yellow)"
    assert summary.loc[summary["grade"] == "ALL", "n"].iloc[0] == 1


def test_parse_grades_orders_and_deduplicates():
    assert parse_grades("Grade_3, Grade_2,Grade_3") == ("Grade_3", "Grade_2")
    assert parse_grades(None) == ()
    with pytest.raises(ValueError, match="at least one"):
        parse_grades(" , ")


def test_annotator_agreement_and_summary(tmp_path: Path):
    first, second = tmp_path / "ann1", tmp_path / "ann2"
    first.mkdir()
    second.mkdir()
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    half = [[2, 2], [2, 10], [10, 10], [10, 2]]
    half[2:] = [[6, 10], [6, 2]]
    _json(first / "a.json", "Yellow_area", box)
    _json(first / "b.json", "Yellow_area", box)
    _json(second / "a.json", "Yellow_area", box)
    _json(second / "b.json", "Yellow_area", half)
    _json(second / "extra.json", "Yellow_area", box)

    agreement = evaluate_annotator_agreement(str(first), str(second))
    assert list(agreement["filename"]) == ["a.json", "b.json"]
    assert agreement["iou"].iloc[0] == 1.0
    assert agreement["iou"].iloc[1] < 1.0
    summary = summarize_agreement(agreement)
    assert summary["n"].iloc[0] == 2
    assert summary["mean_iou"].iloc[0] < 1.0
    assert "share_iou_ge_0.5" in summary.columns


def test_annotator_agreement_rejects_disjoint_sets(tmp_path: Path):
    first, second = tmp_path / "ann1", tmp_path / "ann2"
    first.mkdir()
    second.mkdir()
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    _json(first / "a.json", "Yellow_area", box)
    _json(second / "b.json", "Yellow_area", box)
    with pytest.raises(ValueError, match="No matching sample stems"):
        evaluate_annotator_agreement(str(first), str(second))
    with pytest.raises(ValueError, match="different"):
        evaluate_annotator_agreement(str(first), str(first))


def test_cli_writes_agreement_report_and_sheet(tmp_path: Path):
    pred, gt = _graded_pred_bundle(tmp_path)
    second = tmp_path / "gt2"
    second.mkdir()
    box = [[2, 2], [2, 10], [10, 10], [10, 2]]
    for name in ("white.json", "light.json", "yellow.json"):
        _json(second / name, "Yellow_area", box)
    summary_csv = tmp_path / "summary.csv"
    excel = tmp_path / "report.xlsx"
    evaluate_main([
        "--pred-dir", str(pred),
        "--gt-dir", str(gt),
        "--grade-from", "pred",
        "--grades", "Grade_2,Grade_3",
        "--gt2-dir", str(second),
        "--output-csv", str(tmp_path / "eval.csv"),
        "--output-summary", str(summary_csv),
        "--excel", str(excel),
    ])
    agreement = pd.read_csv(tmp_path / "summary_agreement.csv")
    samples = pd.read_csv(tmp_path / "summary_agreement_samples.csv")
    assert agreement["n"].iloc[0] == len(samples) == 3
    sheets = pd.read_excel(excel, sheet_name=None)
    assert "Annotator_Agreement" in sheets
    grades = pd.read_csv(tmp_path / "summary_by_grade.csv")
    assert set(grades["grade"]) == {"Grade_2", "Grade_3", "ALL"}
