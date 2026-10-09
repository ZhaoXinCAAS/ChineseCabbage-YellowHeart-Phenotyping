r"""Segmentation accuracy evaluation (Reviewer-1 Q1/Q8, Reviewer-2 Q4).

Compares predicted `Yellow_area` masks against independent ground-truth
annotations (same file stems in two folders) and reports IoU / Dice per image,
mask discrepancies requiring human review, and error rates stratified by
independently assigned `simple / typical / hard / blurry` labels (R1-8).
Ground-truth provenance (R2-4) is recorded explicitly: every row keeps
`gt_source` (default `manual_mask`) so the paper can state whether the
number came from hand-drawn masks, sensory grades, or physicochemical assays
(the latter two are covered by `validate_thresholds` instead).

Usage
-----
.. code-block:: bash
    uv run evaluate-segmentation --pred-dir ./pred_json --gt-dir ./gt_json  # demo
    uv run evaluate-segmentation --pred-dir ./pred --gt-dir ./manual_gt
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
import pandas as pd

from yheart._typing import AnnotationFile
from yheart.imaging.masks import read_mask

DEFAULT_PRED_DIR = "./samples_images/results/Auto_segment_Yellow_heart/json"
FAILURE_IOU = 0.5
TINY_PX = 1000

STRATA = ("simple", "typical", "hard", "blurry")


def round_and_clip(points: Sequence[Sequence[float]], w: int, h: int):
    """Round polygon vertices to integer pixels and clamp to the canvas."""
    pts: list[tuple[int, int]] = []
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            continue
        pts.append((max(0, min(w - 1, round(p[0]))), max(0, min(h - 1, round(p[1])))))
    if not pts:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def mask_from_json(path: str | Path, label: str = "Yellow_area"):
    """Rasterize all polygons with ``label`` to a boolean mask."""
    data: AnnotationFile = json.loads(Path(path).read_text(encoding="utf-8"))
    h = data.get("imageHeight", 0)
    w = data.get("imageWidth", 0)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions in {path}: {h} x {w}")
    if label == "Yellow_area" and (raster := data.get("yellow_mask_path")):
        return read_mask(Path(path).parent / raster, (h, w))
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        poly = round_and_clip(shape.get("points", []), w, h)
        if poly.shape[0] >= 3:
            cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], 1)
    return mask.astype(bool)


def iou_dice(pred: np.ndarray, gt: np.ndarray):
    """Return (iou, dice, inter, pred_n, gt_n) with empty-empty => 1.0."""
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    pred_n, gt_n = int(pred.sum()), int(gt.sum())
    if pred_n == 0 and gt_n == 0:
        return 1.0, 1.0, 0, 0, 0
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    iou = inter / union if union else 0.0
    dice = 2 * inter / (pred_n + gt_n) if (pred_n + gt_n) else 0.0
    return iou, dice, inter, pred_n, gt_n


def classify_cause(iou: float, pred_n: int, gt_n: int, failure_iou: float = FAILURE_IOU):
    """Describe mask discrepancies; causal attribution requires human review."""
    if pred_n == 0 and gt_n == 0:
        return "both-empty"
    if pred_n == 0:
        return "miss-no-pred:review-required"
    if gt_n == 0:
        return "false-positive-empty-gt:review-required"
    if iou >= failure_iou:
        return "ok"
    ratio = pred_n / max(gt_n, 1)
    if ratio < 0.5:
        return "under-segment:review-required"
    if ratio > 2.0:
        return "over-segment:review-required"
    return "boundary-mismatch:review-required"


def auto_stratum(gt_n: int, pred_n: int, iou: float, override: str | None = None):
    """Only accept independently assigned strata, never derive them from IoU."""
    if override in STRATA:
        return override
    return "unassigned"


def evaluate_dirs(
    pred_dir: str,
    gt_dir: str,
    pred_label: str = "Yellow_area",
    gt_label: str = "Yellow_area",
    strata_csv: str | None = None,
    failure_iou: float = FAILURE_IOU,
    gt_source: str = "manual_mask",
):
    """Use all GT samples, count absent predictions, reject coordinate drift."""
    if Path(pred_dir).resolve() == Path(gt_dir).resolve():
        raise ValueError("Prediction and ground-truth directories must be different")
    if not Path(pred_dir).is_dir() or not Path(gt_dir).is_dir():
        raise ValueError("Prediction and ground-truth directories must exist")
    if not 0 <= failure_iou <= 1:
        raise ValueError("failure_iou must lie in [0, 1]")
    strata_map: dict[str, str] = {}
    if strata_csv:
        sdf = pd.read_csv(strata_csv)
        cols = {c.lower(): c for c in sdf.columns}
        s_col = cols.get("stratum", cols.get("strata"))
        if "filename" not in cols or not isinstance(s_col, str):
            raise ValueError("Strata CSV requires filename,stratum columns")
        for _, r in sdf.iterrows():
            stem = Path(str(r[cols["filename"]]).strip()).stem
            value = str(r[s_col]).strip()
            if stem in strata_map or value not in STRATA:
                raise ValueError(f"Duplicate sample or invalid stratum: {stem}: {value}")
            strata_map[stem] = value
    pred_files = {p.stem: p for p in Path(pred_dir).glob("*.json")}
    gt_files = {p.stem: p for p in Path(gt_dir).glob("*.json")}
    rows: list[dict[str, object]] = []
    if extras := sorted(set(pred_files) - set(gt_files)):
        raise ValueError(f"Predictions without ground truth: {extras[:10]}")
    for stem in sorted(gt_files):
        gt = mask_from_json(gt_files[stem], gt_label)
        missing = stem not in pred_files
        pred = np.zeros_like(gt) if missing else mask_from_json(pred_files[stem], pred_label)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {stem}: {pred.shape} vs {gt.shape}")
        iou, dice, inter, pred_n, gt_n = iou_dice(pred, gt)
        if missing:
            iou, dice = 0.0, 0.0
        fname = f"{stem}.json"
        rows.append({
            "filename": fname,
            "gt_source": gt_source,
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "inter_px": inter,
            "pred_px": pred_n,
            "gt_px": gt_n,
            "failure": missing or iou < failure_iou,
            "cause": "missing-prediction-file"
            if missing
            else classify_cause(iou, pred_n, gt_n, failure_iou),
            "prediction_missing": missing,
            "area_error_px": pred_n - gt_n,
            "relative_area_error": (pred_n - gt_n) / gt_n if gt_n else float("nan"),
            "stratum": auto_stratum(gt_n, pred_n, iou, strata_map.get(stem)),
        })
    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "gt_source",
            "iou",
            "dice",
            "inter_px",
            "pred_px",
            "gt_px",
            "failure",
            "cause",
            "prediction_missing",
            "area_error_px",
            "relative_area_error",
            "stratum",
        ],
    )


def summarize(df: pd.DataFrame):
    """Aggregate per-image scores into stratum means and failure rates."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("stratum")
    out = (
        g
        .agg(
            n=("iou", "size"),
            mean_iou=("iou", "mean"),
            mean_dice=("dice", "mean"),
            failure_rate=("failure", "mean"),
        )
        .round(4)
        .reset_index()
    )
    all_row = pd.DataFrame([
        {
            "stratum": "ALL",
            "n": len(df),
            "mean_iou": round(df["iou"].mean(), 4),
            "mean_dice": round(df["dice"].mean(), 4),
            "failure_rate": round(df["failure"].mean(), 4),
        }
    ])
    return pd.concat([out, all_row], ignore_index=True)


@dataclass(frozen=True, slots=True)
class EvaluateSegmentationArgs:
    """Parsed command-line arguments."""

    pred_dir: str
    gt_dir: str
    pred_label: str
    gt_label: str
    strata: str | None
    output_csv: str
    output_summary: str
    failure_iou: float
    gt_source: str


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Evaluate Yellow_area segmentation.")
    p.add_argument("--pred-dir", default=DEFAULT_PRED_DIR)
    p.add_argument("--gt-dir", required=True, help="Independent manual segmentation annotations.")
    p.add_argument("--pred-label", default="Yellow_area")
    p.add_argument("--gt-label", default="Yellow_area")
    p.add_argument(
        "--strata", default=None, help="CSV with filename,stratum (simple/typical/hard/blurry)."
    )
    p.add_argument("--output-csv", default="./seg_eval.csv")
    p.add_argument("--output-summary", default="./seg_eval_summary.csv")
    p.add_argument("--failure-iou", type=float, default=FAILURE_IOU)
    p.add_argument("--gt-source", default="manual_mask")
    return cast(EvaluateSegmentationArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    if not Path(args.pred_dir).is_dir() or not Path(args.gt_dir).is_dir():
        print(f"Error: pred-dir or gt-dir not found: {args.pred_dir}, {args.gt_dir}")
        sys.exit(1)
    df = evaluate_dirs(
        args.pred_dir,
        args.gt_dir,
        args.pred_label,
        args.gt_label,
        args.strata,
        args.failure_iou,
        args.gt_source,
    )
    if df.empty:
        print("Error: no ground-truth JSON samples found.")
        sys.exit(1)
    summary = summarize(df)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary.to_csv(args.output_summary, index=False)
    iou: "pd.Series[float]" = df["iou"]
    dice: "pd.Series[float]" = df["dice"]
    failure: "pd.Series[bool]" = df["failure"]
    mean_iou = iou.mean()
    mean_dice = dice.mean()
    n_fail = failure.sum()
    print(
        f"Evaluated n={len(df)}: mean IoU={mean_iou:.3f}, "
        f"mean Dice={mean_dice:.3f}, "
        f"failures(IoU<{args.failure_iou})={n_fail}"
    )
    print(summary.to_string(index=False))
    print(f"Saved: {args.output_csv}, {args.output_summary}")


if __name__ == "__main__":
    main(sys.argv[1:])
