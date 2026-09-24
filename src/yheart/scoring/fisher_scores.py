"""Compute Fisher discriminant scores for yellow-heart color features.

For each of the ten color features, this script evaluates its power to
discriminate between three yellow-heart phenotype categories
(``Light_yellow``, ``yellow``, ``Deep_yellow``) using the Fisher discriminant
ratio (between-class variance / within-class variance), determines the
response direction (positive vs. negative correlation with the deep-yellow
class), and exports normalised signed weights.

Usage
-----
.. code-block:: bash
    uv run fisher-scores  # demo dataset (defaults)
    uv run fisher-scores \\
        --features ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/\
roi_10_color_features_with_ratio.xlsx \\
        --labeled-root ./data/Fisher_score_data/ \\
        --output ./data/fisher_weights_418.xlsx  # Figshare full dataset
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Default paths for Demo dataset
DEFAULT_ROI_FEATURES_FILE = "./samples_images/results/roi_10_color_features_with_ratio.xlsx"
DEFAULT_LABELED_ROOT = "./samples_images/Fisher_score_data/"
DEFAULT_OUTPUT_METRICS_EXCEL = "./samples_images/results/fisher_weights_418.xlsx"


FEATURES = [
    "H_circular_mean_deg",
    "S_mean",
    "V_mean",
    "L_mean",
    "a_mean",
    "b_mean",
    "R_mean",
    "G_mean",
    "B_mean",
    "ExR_mean",
]
LABEL_ORDER = ["Light_yellow", "yellow", "Deep_yellow"]


def get_label(filename: str | float | None, labeled_root: str) -> str | None:
    """Map an image filename to its phenotype label by folder membership."""
    if pd.isna(filename):
        return None
    fn = str(filename).strip()
    for label in LABEL_ORDER:
        folder = Path(labeled_root) / label
        if not folder.exists():
            continue
        if (folder / fn).exists():
            return label
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            if (folder / f"{fn}{ext}").exists():
                return label
    return None


def compute_fisher_scores(features_file: str, labeled_root: str, output_file: str):
    """Compute Fisher ratios, response directions, and signed weights."""
    # 1. Load data and map labels
    df_roi = pd.read_excel(features_file)
    df_roi["label"] = df_roi["filename"].apply(lambda fn: get_label(fn, labeled_root))
    df_fisher = df_roi.dropna(subset=["label"]).copy()
    if df_fisher.empty:
        print(f"Error: No labeled samples found under '{labeled_root}'.")
        return

    # 2. Normalization
    X: np.ndarray = np.asarray(MinMaxScaler().fit_transform(df_fisher[FEATURES].values))
    y = np.asarray(df_fisher["label"].values)

    # 3. Calculate Fisher Scores
    total_mean = np.mean(X, axis=0)
    classes = np.unique(y)
    fisher_dict: dict[str, float] = {}
    for i, feat in enumerate(FEATURES):
        num, den = 0.0, 0.0
        for cls in classes:
            cls_feat = X[y == cls, i]
            n_k = len(cls_feat)
            if n_k > 0:
                num += n_k * ((np.mean(cls_feat) - total_mean[i]) ** 2)
                den += n_k * np.var(cls_feat)
        fisher_dict[feat] = num / den if den >= 1e-10 else 0.0

    # 4. Determine response direction and normalized weights
    mask_light, mask_deep = (y == LABEL_ORDER[0]), (y == LABEL_ORDER[2])
    dir_dict = {
        feat: (
            "Positive (+)"
            if np.mean(X[mask_deep, i]) > np.mean(X[mask_light, i])
            else "Negative (-)"
        )
        for i, feat in enumerate(FEATURES)
    }
    scores_arr = np.array([fisher_dict[f] for f in FEATURES])
    norm_weights = scores_arr / scores_arr.sum()

    # 5. Build results DataFrame
    records: list[dict[str, object]] = []
    for i, feat in enumerate(FEATURES):
        direction_sign = 1 if "Positive" in dir_dict[feat] else -1
        records.append({
            "Feature_Name": feat,
            "Fisher_Score": round(fisher_dict[feat], 4),
            "Response_Direction": dir_dict[feat],
            "Final_Weight": round(norm_weights[i] * direction_sign, 4),
        })
    df_result = pd.DataFrame(records)
    df_result = df_result.sort_values(by="Fisher_Score", ascending=False).reset_index(drop=True)

    # 6. Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_result.to_excel(output_file, index=False)
    print(f"Fisher score calculation complete. Output saved to: {output_file}")


@dataclass(frozen=True, slots=True)
class FisherScoresArgs:
    """Parsed command-line arguments."""

    features: str
    labeled_root: str
    output: str


def parse_args(argv: Sequence[str] | None = None) -> FisherScoresArgs:
    p = argparse.ArgumentParser(
        description="Compute Fisher discriminant scores for yellow-heart color features."
    )
    p.add_argument(
        "--features", default=DEFAULT_ROI_FEATURES_FILE, help="Input .xlsx of raw color features."
    )
    p.add_argument(
        "--labeled-root",
        default=DEFAULT_LABELED_ROOT,
        help="Root folder of labeled phenotype subfolders.",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_METRICS_EXCEL, help="Output .xlsx of Fisher weights."
    )
    return cast(FisherScoresArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    compute_fisher_scores(args.features, args.labeled_root, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
