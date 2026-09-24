r"""Baseline comparison for the FDA weighting (Reviewer-1 Q5).

Answers "why FDA instead of RF / SVM / modern ML" with numbers: the same
10 color features and the same three phenotype labels
(`Light_yellow / yellow / Deep_yellow`) are fed to four classifiers under
identical stratified k-fold cross-validation — LDA (the multi-class Fisher
discriminant itself), logistic regression, random forest, and RBF-SVM — and
the resulting accuracy / macro-F1 / confusion matrices are exported for the
rebuttal table.

Like `fisher_scores.py`, labels come from folder membership under
`--labeled-root`; MinMax scaling is fitted *inside* each fold (no leakage).
With no labeled samples the script prints a diagnostic and exits cleanly.

Usage
-----
.. code-block:: bash
    uv run baseline-compare \
        --features ./data/.../roi_10_color_features_with_ratio.xlsx \
        --labeled-root ./data/Fisher_score_data/ \
        --output ./data/baseline_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from yheart.scoring.fisher_scores import (
    FEATURES,
    LABEL_ORDER,
    get_label,
)

type Estimator = LinearDiscriminantAnalysis | LogisticRegression | RandomForestClassifier | SVC

DEFAULT_FEATURES = "./samples_images/results/roi_10_color_features_with_ratio.xlsx"
DEFAULT_LABELED_ROOT = "./samples_images/Fisher_score_data/"
DEFAULT_OUTPUT = "./samples_images/results/baseline_metrics.csv"
DEFAULT_CONFUSION = "./samples_images/results/baseline_confusion.csv"
DEFAULT_SPLITS = 5
DEFAULT_SEED = 42


def build_models(random_state: int = DEFAULT_SEED) -> Mapping[str, Estimator]:
    """Compare the four classifiers under identical CV folds."""
    return {
        "LDA(FDA)": LinearDiscriminantAnalysis(),
        "LogReg": LogisticRegression(max_iter=2000, random_state=random_state),
        "RF": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "SVM-RBF": SVC(kernel="rbf", random_state=random_state),
    }


def load_labeled(features_file: str, labeled_root: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Join feature rows with folder-membership phenotype labels."""
    df = pd.read_excel(features_file)
    df["label"] = df["filename"].apply(lambda fn: get_label(fn, labeled_root))
    df = df.dropna(subset=["label"]).copy()
    assert isinstance(df, pd.DataFrame)
    if df.empty:
        return None
    if missing := [f for f in FEATURES if f not in df.columns]:
        raise KeyError(f"Feature columns missing: {missing}")
    return df[FEATURES].to_numpy(dtype=float), np.asarray(df["label"].values)


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    models: Mapping[str, Estimator],
    n_splits: int = DEFAULT_SPLITS,
    random_state: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified k-fold per model; scaler fitted inside each fold."""
    labels = list(LABEL_ORDER)
    unique, counts = np.unique(y, return_counts=True)
    count_of = dict(zip(unique.tolist(), counts.tolist()))
    present_counts = [n for c in labels if (n := count_of.get(c, 0)) > 0]
    n_splits = max(2, min(n_splits, min(present_counts)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metric_rows: list[dict[str, object]] = []
    conf_rows: list[dict[str, object]] = []
    for name, clf in models.items():
        y_true_all: list[str] = []
        y_pred_all: list[str] = []
        for train, test in skf.split(X, y):
            scaler = MinMaxScaler().fit(X[train])
            clf.fit(scaler.transform(X[train]), y[train])
            y_true_all.extend(str(v) for v in y[test].tolist())
            y_pred_all.extend(str(v) for v in clf.predict(scaler.transform(X[test])))
        y_true_arr = np.asarray(y_true_all)
        y_pred_arr = np.asarray(y_pred_all)
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
        row: dict[str, object] = {
            "model": name,
            "accuracy": round(accuracy_score(y_true_arr, y_pred_arr), 4),
            "macro_f1": round(
                float(
                    f1_score(
                        y_true_arr,
                        y_pred_arr,
                        average="macro",
                        # sklearn runtime accepts 0/1/"warn"; stubs type it as str.
                        zero_division=0,  # pyright: ignore[reportArgumentType]
                    )
                ),
                4,
            ),
            "n_splits": n_splits,
            "n_samples": len(y),
        }
        for j, cls in enumerate(labels):
            denom = cm[j].sum()
            row[f"recall_{cls}"] = round(cm[j, j] / denom, 4) if denom else 0.0
        metric_rows.append(row)
        for a, true in enumerate(labels):
            conf_rows.extend(
                {
                    "model": name,
                    "true": true,
                    "pred": pred,
                    "count": int(cm[a, b]),
                }
                for b, pred in enumerate(labels)
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(conf_rows)


def run(
    features_file: str,
    labeled_root: str,
    output: str,
    confusion_out: str,
    n_splits: int,
    random_state: int,
) -> bool:
    """Run the comparison; return False when labels are absent."""
    try:
        loaded = load_labeled(features_file, labeled_root)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return False
    if loaded is None:
        print(
            f"No labeled samples found under '{labeled_root}'. "
            "Baseline comparison skipped (needs phenotype-label subfolders)."
        )
        return False
    X, y = loaded
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        print(
            "Not enough labeled samples per class for cross-validation "
            f"(classes={sorted(map(str, np.unique(y)))}). Skipped."
        )
        return False
    metrics, conf = cross_validate(X, y, build_models(random_state), n_splits, random_state)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    conf.to_csv(confusion_out, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved: {output}, {confusion_out}")
    return True


@dataclass(frozen=True, slots=True)
class BaselineCompareArgs:
    """Parsed command-line arguments."""

    features: str
    labeled_root: str
    output: str
    confusion: str
    n_splits: int
    random_state: int


def parse_args(argv: Sequence[str] | None = None) -> BaselineCompareArgs:
    p = argparse.ArgumentParser(
        description="FDA vs RF/SVM baseline comparison (stratified k-fold)."
    )
    p.add_argument("--features", default=DEFAULT_FEATURES)
    p.add_argument("--labeled-root", default=DEFAULT_LABELED_ROOT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--confusion", default=DEFAULT_CONFUSION)
    p.add_argument("--n-splits", type=int, default=DEFAULT_SPLITS)
    p.add_argument("--random-state", type=int, default=DEFAULT_SEED)
    return cast(BaselineCompareArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    run(
        args.features,
        args.labeled_root,
        args.output,
        args.confusion,
        args.n_splits,
        args.random_state,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
