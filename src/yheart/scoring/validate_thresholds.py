r"""External validation of GMM grade thresholds (Reviewer-1 Q3/Q7).

The five CYS grades are data-driven; this module validates them against
independent references without requiring new experiments in the repo:

* expert / consumer-panel sensory grades (R1-3): accuracy, quadratic-weighted
  Cohen's κ (ordinal grades), per-grade precision/recall, confusion matrix;
* HPLC carotenoid content on grade-spanning accessions (R1-7): Spearman ρ and
  Pearson r between CYS (or Yellow_score) and carotenoid concentration.

Each analysis runs only when its reference column is present; otherwise the
script prints what is missing and exits cleanly.

Usage
-----
.. code-block:: bash
    uv run validate-thresholds \
        --graded ./data/Plot/CYS_1319_GMM_Final_5Classes.xlsx \
        --ref ./data/external_panel.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

DEFAULT_GRADED = "./samples_images/results/Plot/CYS_1319_GMM_Final_5Classes.xlsx"
DEFAULT_REF = "./samples_images/external_reference.csv"
DEFAULT_JOIN = "filename"
DEFAULT_PRED_GRADE = "Scientific_Grade"
DEFAULT_TRUE_GRADE = "expert_grade"
DEFAULT_SCORE = "CYS"
DEFAULT_HPLC = "carotenoid"


def read_table(path: str):
    """Read a CSV/XLSX reference table."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported reference format (need .csv/.xlsx): {path}")


def coerce_ordinal(df: pd.DataFrame, col: str):
    """Map a grade column to ints when possible (keeps quadratic-κ ordering)."""
    raw = df[col]
    assert isinstance(raw, pd.Series), f"{col} must be a single column"

    # Quoted: pandas Series is not subscriptable at runtime.
    numeric: "pd.Series[float]" = pd.to_numeric(raw, errors="coerce")
    if bool(numeric.notna().all()):
        return numeric.astype(int)
    return raw.astype(str).str.strip()


def grade_agreement(df: pd.DataFrame, pred_col: str, true_col: str) -> dict[str, object]:
    """Accuracy, quadratic kappa, and per-grade precision/recall vs expert grades."""
    sub = df[[pred_col, true_col]].dropna().copy()
    assert isinstance(sub, pd.DataFrame)
    if sub.empty:
        raise ValueError("No overlapping graded samples for agreement.")
    pred = coerce_ordinal(sub, pred_col)
    true = coerce_ordinal(sub, true_col)
    labels = sorted(set(pred) | set(true))
    cm = pd.crosstab(true, pred, rownames=["true"], colnames=["pred"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    assert isinstance(cm, pd.DataFrame)
    per_grade: dict[str, dict[str, float]] = {}
    mat = cm.to_numpy()
    for i, lab in enumerate(labels):
        col_sum = mat[:, i].sum()
        row_sum = mat[i, :].sum()
        tp = mat[i, i]
        per_grade[str(lab)] = {
            "precision": round(tp / col_sum, 4) if col_sum else 0.0,
            "recall": round(tp / row_sum, 4) if row_sum else 0.0,
        }
    return {
        "n": len(sub),
        "accuracy": round(np.mean(pred.to_numpy() == true.to_numpy()), 4),
        "quadratic_kappa": round(
            cohen_kappa_score(true, pred, labels=labels, weights="quadratic"), 4
        ),
        "per_grade": per_grade,
        "confusion": cm,
    }


def hplc_correlation(df: pd.DataFrame, score_col: str, hplc_col: str) -> dict[str, object]:
    """Spearman and Pearson correlation between CYS and HPLC carotenoid content."""
    sub = df[[score_col, hplc_col]].dropna().copy()
    assert isinstance(sub, pd.DataFrame)
    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
    sub[hplc_col] = pd.to_numeric(sub[hplc_col], errors="coerce")
    sub = sub.dropna()
    assert isinstance(sub, pd.DataFrame)
    if len(sub) < 3:
        raise ValueError("Need >=3 samples with both score and HPLC values.")
    spearman_rho, spearman_p = (cast(float, v) for v in spearmanr(sub[score_col], sub[hplc_col]))
    pearson_r, pearson_p = (cast(float, v) for v in pearsonr(sub[score_col], sub[hplc_col]))
    return {
        "n": len(sub),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": spearman_p,
        "pearson_r": round(pearson_r, 4),
        "pearson_p": pearson_p,
    }


def _add_grade_metrics(
    df: pd.DataFrame,
    pred_grade: str,
    true_grade: str,
    metrics: dict[str, object],
    confusion_out: str,
) -> bool:
    """Add grade-agreement metrics when the required columns are available."""
    if true_grade not in df.columns or pred_grade not in df.columns:
        print(f"Grade agreement skipped: need '{pred_grade}' + '{true_grade}'.")
        return False

    try:
        agree = grade_agreement(df, pred_grade, true_grade)
    except ValueError as e:
        print(f"Grade agreement skipped: {e}")
        return False

    metrics |= {
        "accuracy_vs_expert": agree["accuracy"],
        "quadratic_kappa_vs_expert": agree["quadratic_kappa"],
        "n_agreement": agree["n"],
    }
    for g, pr in cast(dict[str, dict[str, float]], agree["per_grade"]).items():
        metrics[f"precision_grade_{g}"] = pr["precision"]
        metrics[f"recall_grade_{g}"] = pr["recall"]

    cast(pd.DataFrame, agree["confusion"]).to_csv(confusion_out)
    print(f"Confusion matrix saved to: {confusion_out}")
    return True


def _add_hplc_metrics(
    df: pd.DataFrame,
    score_col: str,
    hplc_col: str,
    metrics: dict[str, object],
) -> bool:
    """Add HPLC correlation metrics when the required columns are available."""
    if hplc_col not in df.columns or score_col not in df.columns:
        print(f"HPLC correlation skipped: need '{score_col}' + '{hplc_col}'.")
        return False

    try:
        corr = hplc_correlation(df, score_col, hplc_col)
    except ValueError as e:
        print(f"HPLC correlation skipped: {e}")
        return False

    metrics |= {
        "hplc_spearman_rho": corr["spearman_rho"],
        "hplc_spearman_p": corr["spearman_p"],
        "hplc_pearson_r": corr["pearson_r"],
        "hplc_pearson_p": corr["pearson_p"],
        "n_hplc": corr["n"],
    }
    return True


def run(
    graded_path: str,
    ref_path: str,
    join_col: str,
    pred_grade: str,
    true_grade: str,
    score_col: str,
    hplc_col: str,
    metrics_out: str,
    confusion_out: str,
):
    """Run external validation; return False when references are absent."""
    try:
        graded = read_table(graded_path)
        ref = read_table(ref_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return False

    if join_col not in graded.columns or join_col not in ref.columns:
        print(f"Error: join column '{join_col}' must exist in both tables.")
        return False

    df = graded.merge(ref, on=join_col, suffixes=("", "_ref"))
    if df.empty:
        print("Warning: no overlapping filenames between graded and reference.")
        return False

    metrics: dict[str, object] = {"n_overlap": len(df)}
    did_anything = _add_grade_metrics(df, pred_grade, true_grade, metrics, confusion_out)
    did_anything = _add_hplc_metrics(df, score_col, hplc_col, metrics) or did_anything

    if not did_anything:
        return False

    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved: {metrics_out}")
    return True


@dataclass(frozen=True, slots=True)
class ValidateThresholdsArgs:
    """Parsed command-line arguments."""

    graded: str
    ref: str
    on: str
    pred_grade: str
    true_grade: str
    score: str
    hplc: str
    output: str
    confusion: str


def parse_args(argv: Sequence[str] | None = None) -> ValidateThresholdsArgs:
    p = argparse.ArgumentParser(description="Validate GMM thresholds vs expert grades / HPLC.")
    p.add_argument("--graded", default=DEFAULT_GRADED)
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--on", default=DEFAULT_JOIN)
    p.add_argument("--pred-grade", default=DEFAULT_PRED_GRADE)
    p.add_argument("--true-grade", default=DEFAULT_TRUE_GRADE)
    p.add_argument("--score", default=DEFAULT_SCORE)
    p.add_argument("--hplc", default=DEFAULT_HPLC)
    p.add_argument("--output", default="./validation_metrics.csv")
    p.add_argument("--confusion", default="./validation_confusion.csv")
    return cast(ValidateThresholdsArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    run(
        args.graded,
        args.ref,
        args.on,
        args.pred_grade,
        args.true_grade,
        args.score,
        args.hplc,
        args.output,
        args.confusion,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
