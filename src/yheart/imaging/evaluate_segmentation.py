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
    uv run evaluate-segmentation --pred-dir ./ByGrade_5_100 --gt-dir ./annotations \\
        --grade-from pred --grades Grade_2,Grade_3,Grade_4,Grade_5
    uv run evaluate-segmentation --pred-dir ./pred --gt-dir ./annotator_1 \\
        --gt2-dir ./annotator_2 --output-summary ./summary.csv
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

#: Display names used by the paper's per-grade benchmark table. Grades absent
#: from this map keep their folder name so an unexpected folder is never
#: silently mislabelled.
GRADE_LABELS: dict[str, str] = {
    "Grade_2": "Grade 2 (Light Yellow)",
    "Grade_3": "Grade 3 (Yellow)",
    "Grade_4": "Grade 4 (Deep Yellow)",
    "Grade_5": "Grade 5 (Extreme Yellow)",
}

GRADE_SOURCES = ("gt", "pred", "none")


def parse_grades(value: str | None) -> tuple[str, ...]:
    """Split ``--grades Grade_2,Grade_3`` into an ordered tuple of folder names."""
    if not value:
        return ()
    grades = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not grades:
        raise ValueError("--grades must name at least one grade folder")
    return grades


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
    gt_subdirs: bool = False,
    grade_from: str | None = None,
    grades: Sequence[str] | None = None,
):
    """Use all GT samples, count absent predictions, reject coordinate drift.

    The ``grade`` column is filled from the immediate folder name of either
    tree:

    * ``grade_from="gt"`` (requires ``gt_subdirs=True``) — the graded
      ground-truth layout ``gt/Grade_2/*.json``;
    * ``grade_from="pred"`` — the graded prediction layout used by the
      revision benchmark (``pred/Grade_2/*.json`` with flat ground truth);
      the prediction tree is scanned recursively and defines the evaluated
      set, so extra ground-truth samples outside it are ignored;
    * ``grade_from="none"``/``None`` — no grade information.

    ``grades`` keeps only the named folders; an unknown or empty grade name
    raises instead of silently producing a short table.
    """
    if Path(pred_dir).resolve() == Path(gt_dir).resolve():
        raise ValueError("Prediction and ground-truth directories must be different")
    if not Path(pred_dir).is_dir() or not Path(gt_dir).is_dir():
        raise ValueError("Prediction and ground-truth directories must exist")
    if not 0 <= failure_iou <= 1:
        raise ValueError("failure_iou must lie in [0, 1]")
    source = grade_from or ("gt" if gt_subdirs else "none")
    if source not in GRADE_SOURCES:
        raise ValueError(f"grade_from must be one of {GRADE_SOURCES}, got {source!r}")
    if source == "gt" and not gt_subdirs:
        raise ValueError("grade_from='gt' requires gt_subdirs=True (a graded GT tree)")
    allowed = tuple(grades or ())
    if allowed and source == "none":
        raise ValueError("--grades needs grade information: pass --gt-subdirs or --grade-from pred")
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
    pred_root = Path(pred_dir)
    pred_grades: dict[str, str] = {}
    if source == "pred":
        pred_files: dict[str, Path] = {}
        for path in sorted(pred_root.rglob("*.json")):
            if path.stem in pred_files:
                raise ValueError(f"Duplicate prediction sample across grade folders: {path.stem}")
            pred_files[path.stem] = path
            pred_grades[path.stem] = path.parent.name
    else:
        pred_files = {p.stem: p for p in pred_root.glob("*.json")}
    if gt_subdirs:
        gt_files: dict[str, Path] = {}
        gt_root = Path(gt_dir)
        for path in sorted(gt_root.rglob("*.json")):
            if path.stem in gt_files:
                raise ValueError(
                    f"Duplicate ground-truth sample across grade folders: {path.stem}"
                )
            gt_files[path.stem] = path
    else:
        gt_files = {p.stem: p for p in Path(gt_dir).glob("*.json")}
    if source == "pred":
        # The graded prediction tree defines the evaluated set, so a shared GT
        # folder holding samples outside the benchmark is not an error.
        gt_files = {stem: path for stem, path in gt_files.items() if stem in pred_files}
    rows: list[dict[str, object]] = []
    if extras := sorted(set(pred_files) - set(gt_files)):
        raise ValueError(f"Predictions without ground truth: {extras[:10]}")
    for stem in sorted(gt_files):
        grade = ""
        if source == "gt":
            grade = gt_files[stem].parent.name if gt_files[stem].parent != Path(gt_dir) else ""
        elif source == "pred":
            grade = pred_grades.get(stem, "")
        if allowed and grade not in allowed:
            continue
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
            "grade": grade,
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
    table = pd.DataFrame(
        rows,
        columns=[
            "filename",
            "gt_source",
            "grade",
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
    if allowed:
        seen = {str(value) for value in table["grade"]}
        if unknown := [grade for grade in allowed if grade not in seen]:
            raise ValueError(f"Requested grades have no samples: {unknown}; found {sorted(seen)}")
    return table


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


def _format_mean_sd(values: "pd.Series[float]") -> str:
    """Paper-table stem: ``mean ± SD`` with population SD (ddof=0)."""
    if values.empty:
        return "N/A"
    return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"


def summarize_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IoU/Dice by phenotype grade folder, plus an overall row.

    Grades come from the ``grade`` column (``--gt-subdirs`` or
    ``--grade-from pred``). Columns mirror the revision benchmark table:
    sample count, mean/SD, printable ``mean ± SD`` strings, and the display
    label used in the paper table.
    """
    if df.empty or "grade" not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups = [(str(grade), g) for grade, g in df.groupby("grade", sort=True) if grade]
    if not groups:
        groups = [("all", df)]
    for grade, g in groups:
        iou: "pd.Series[float]" = g["iou"]
        dice: "pd.Series[float]" = g["dice"]
        rows.append({
            "grade": grade,
            "grade_label": GRADE_LABELS.get(grade, grade),
            "n": len(g),
            "mean_iou": round(iou.mean(), 4),
            "sd_iou": round(iou.std(ddof=0), 4),
            "mean_dice": round(dice.mean(), 4),
            "sd_dice": round(dice.std(ddof=0), 4),
            "iou_formatted": _format_mean_sd(iou),
            "dice_formatted": _format_mean_sd(dice),
            "failure_rate": round(float(g["failure"].mean()), 4),
        })
    all_iou: "pd.Series[float]" = df["iou"]
    all_dice: "pd.Series[float]" = df["dice"]
    rows.append({
        "grade": "ALL",
        "grade_label": "Overall (all retained grades)",
        "n": len(df),
        "mean_iou": round(all_iou.mean(), 4),
        "sd_iou": round(all_iou.std(ddof=0), 4),
        "mean_dice": round(all_dice.mean(), 4),
        "sd_dice": round(all_dice.std(ddof=0), 4),
        "iou_formatted": _format_mean_sd(all_iou),
        "dice_formatted": _format_mean_sd(all_dice),
        "failure_rate": round(float(df["failure"].mean()), 4),
    })
    return pd.DataFrame(rows)


def evaluate_annotator_agreement(
    gt1_dir: str,
    gt2_dir: str,
    label1: str = "Yellow_area",
    label2: str = "Yellow_area",
    failure_iou: float = FAILURE_IOU,
) -> pd.DataFrame:
    """Agreement between two independent annotations of the same samples (R1-1).

    Returns one row per sample stem present in both trees. ``iou``/``dice``
    are computed between the two human masks, i.e. the measurement ceiling
    the algorithm is judged against.
    """
    if Path(gt1_dir).resolve() == Path(gt2_dir).resolve():
        raise ValueError("The two annotation directories must be different")
    first = {p.stem: p for p in Path(gt1_dir).glob("*.json")}
    second = {p.stem: p for p in Path(gt2_dir).glob("*.json")}
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("No matching sample stems between the two annotation sets")
    rows: list[dict[str, object]] = []
    for stem in shared:
        mask_a = mask_from_json(first[stem], label1)
        mask_b = mask_from_json(second[stem], label2)
        if mask_a.shape != mask_b.shape:
            raise ValueError(f"Shape mismatch for {stem}: {mask_a.shape} vs {mask_b.shape}")
        iou, dice, inter, a_px, b_px = iou_dice(mask_a, mask_b)
        rows.append({
            "filename": f"{stem}.json",
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "inter_px": inter,
            "annotator1_px": a_px,
            "annotator2_px": b_px,
            "area_error_px": b_px - a_px,
            "relative_area_error": (b_px - a_px) / a_px if a_px else float("nan"),
            "agreement_ok": iou >= failure_iou,
        })
    return pd.DataFrame(rows)


def summarize_agreement(df: pd.DataFrame, failure_iou: float = FAILURE_IOU) -> pd.DataFrame:
    """One-row summary of inter-annotator IoU/Dice with printable mean ± SD."""
    if df.empty:
        return pd.DataFrame()
    iou: "pd.Series[float]" = df["iou"]
    dice: "pd.Series[float]" = df["dice"]
    return pd.DataFrame([{
        "n": len(df),
        "mean_iou": round(iou.mean(), 4),
        "sd_iou": round(iou.std(ddof=0), 4),
        "mean_dice": round(dice.mean(), 4),
        "sd_dice": round(dice.std(ddof=0), 4),
        "iou_formatted": _format_mean_sd(iou),
        "dice_formatted": _format_mean_sd(dice),
        f"share_iou_ge_{failure_iou:g}": round(float(df["agreement_ok"].mean()), 4),
    }])


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
    gt_subdirs: bool
    output_grade_summary: str | None
    excel: str | None
    grade_from: str | None
    grades: str | None
    gt2_dir: str | None
    gt2_label: str | None
    output_agreement: str | None


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
    p.add_argument(
        "--gt-subdirs",
        action="store_true",
        help="Read GT recursively and report the immediate folder as the phenotype grade.",
    )
    p.add_argument(
        "--output-grade-summary",
        default=None,
        help="CSV for the per-grade table (default: <output-summary>_by_grade.csv).",
    )
    p.add_argument(
        "--excel",
        default=None,
        help="Optional .xlsx workbook with summary and per-sample sheets.",
    )
    p.add_argument(
        "--grade-from",
        choices=["gt", "pred", "none"],
        default=None,
        help="Where the grade label comes from: graded GT folders, graded prediction "
        "folders (scanned recursively), or none. Defaults to gt with --gt-subdirs.",
    )
    p.add_argument(
        "--grades",
        default=None,
        help="Comma-separated grade folders to keep, e.g. Grade_2,Grade_3,Grade_4,Grade_5.",
    )
    p.add_argument(
        "--gt2-dir",
        default=None,
        help="Second independent annotation set; reports inter-annotator IoU/Dice.",
    )
    p.add_argument(
        "--gt2-label",
        default=None,
        help="Label used by the second annotation set (default: same as --gt-label).",
    )
    p.add_argument(
        "--output-agreement",
        default=None,
        help="CSV stem for the agreement report (default: <output-summary>_agreement).",
    )
    return cast(EvaluateSegmentationArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    if not Path(args.pred_dir).is_dir() or not Path(args.gt_dir).is_dir():
        print(f"Error: pred-dir or gt-dir not found: {args.pred_dir}, {args.gt_dir}")
        sys.exit(1)
    grade_source = args.grade_from or ("gt" if args.gt_subdirs else "none")
    try:
        grades = parse_grades(args.grades)
        df = evaluate_dirs(
            args.pred_dir,
            args.gt_dir,
            args.pred_label,
            args.gt_label,
            args.strata,
            args.failure_iou,
            args.gt_source,
            args.gt_subdirs,
            grade_source,
            grades,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
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
    if grades:
        print(f"Grade filter: {', '.join(grades)} -> n={len(df)}")
    print(f"Saved: {args.output_csv}, {args.output_summary}")

    grade_summary = summarize_by_grade(df) if grade_source != "none" else pd.DataFrame()
    if not grade_summary.empty:
        grade_path = args.output_grade_summary or str(
            Path(args.output_summary).with_name(
                f"{Path(args.output_summary).stem}_by_grade.csv"
            )
        )
        Path(grade_path).parent.mkdir(parents=True, exist_ok=True)
        grade_summary.to_csv(grade_path, index=False)
        print("\nPer-grade benchmark:")
        print(grade_summary.to_string(index=False))
        print(f"Saved: {grade_path}")

    agreement_summary = pd.DataFrame()
    agreement: pd.DataFrame | None = None
    if args.gt2_dir:
        try:
            agreement = evaluate_annotator_agreement(
                args.gt_dir,
                args.gt2_dir,
                args.gt_label,
                args.gt2_label or args.gt_label,
                args.failure_iou,
            )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        agreement_summary = summarize_agreement(agreement, args.failure_iou)
        agreement_stem = args.output_agreement or str(
            Path(args.output_summary).with_name(
                f"{Path(args.output_summary).stem}_agreement"
            )
        )
        agreement_samples_path = f"{agreement_stem}_samples.csv"
        Path(agreement_stem).parent.mkdir(parents=True, exist_ok=True)
        agreement_summary.to_csv(f"{agreement_stem}.csv", index=False)
        agreement.to_csv(agreement_samples_path, index=False)
        print("\nInter-annotator agreement:")
        print(agreement_summary.to_string(index=False))
        print(f"Saved: {agreement_stem}.csv, {agreement_samples_path}")

    if args.excel:
        Path(args.excel).parent.mkdir(parents=True, exist_ok=True)
        table = grade_summary if not grade_summary.empty else summary
        with pd.ExcelWriter(args.excel, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name="Table_Summary", index=False)
            df.to_excel(writer, sheet_name="Per_Sample_Details", index=False)
            if not agreement_summary.empty and agreement is not None:
                agreement_summary.to_excel(
                    writer, sheet_name="Annotator_Agreement", index=False
                )
                agreement.to_excel(
                    writer, sheet_name="Annotator_Agreement_Samples", index=False
                )
        print(f"Saved: {args.excel}")


if __name__ == "__main__":
    main(sys.argv[1:])
