"""Reproduce the Fisher-score feature ranking from the labeled ROI cohort.

Adapted from 3_1_Fisher_scoreTabel3.py in the 2026-11-16 study scripts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 4. 10color feature column name
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

# 5. Tag order
LABEL_ORDER = ["yellow_low", "yellow_medium", "yellow_many"]


def get_label_from_filename(filename: object, labeled_root: Path) -> str | None:
    if not isinstance(filename, str) or not filename:
        return None
    for label in LABEL_ORDER:
        if (labeled_root / label / filename).is_file():
            return label
    return None


def fisher_score_multi_class(X, y, feature_names):
    scores = []

    classes = np.unique(y)

    total_mean = np.mean(X, axis=0)

    for i, feat_name in enumerate(feature_names):
        feat = X[:, i]

        numerator = 0.0
        denominator = 0.0

        for cls in classes:
            cls_mask = y == cls

            cls_feat = feat[cls_mask]

            n_k = len(cls_feat)

            if n_k == 0:
                continue

            mu_k = np.mean(cls_feat)

            var_k = np.var(cls_feat)

            numerator += n_k * (mu_k - total_mean[i]) ** 2

            denominator += n_k * var_k

        if denominator < 1e-10:
            score = np.inf
        else:
            score = numerator / denominator

        scores.append((feat_name, score))

    # pressFisher ScoreSort from large to small
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def get_feature_directions(X, y, feature_names, label_order):
    directions = {}

    for i, feat_name in enumerate(feature_names):
        mean_low = np.mean(X[y == label_order[0], i])

        mean_many = np.mean(X[y == label_order[2], i])

        directions[feat_name] = 1 if mean_many > mean_low else -1

    return directions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the study Fisher-score table."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("=" * 60)
    print("ReadingROIFeature data...")

    df_roi = pd.read_excel(args.input)

    print(f"Read in total {len(df_roi)} samples")

    # -------------------------- 2. Automatically match tags --------------------------

    print("\nMatching tags...")

    df_roi["label"] = df_roi["filename"].apply(
        lambda x: get_label_from_filename(x, args.labels_dir)
    )

    # Statistical label distribution
    label_counts = df_roi["label"].value_counts()

    print(f"Label distribution:\n{label_counts}")

    if df_roi["label"].isna().any():
        print(
            f"warning:Yes {df_roi['label'].isna().sum()} " "samples did not match the label,will be excluded"
        )

    # Filter labeled samples
    df_fisher = df_roi.dropna(subset=["label"]).copy()

    missing_labels = set(LABEL_ORDER) - set(df_fisher["label"])
    if missing_labels:
        raise ValueError(
            f"Missing phenotype classes in labeled data: {sorted(missing_labels)}"
        )
    print(f"used forFisherThe number of valid samples analyzed:{len(df_fisher)}")

    # -------------------------- 3. Feature normalization --------------------------

    print("\nStandardizing features...")

    X_fisher = df_fisher[FEATURES].values
    y_fisher = df_fisher["label"].values

    scaler = MinMaxScaler(feature_range=(0, 1))

    X_fisher_scaled = scaler.fit_transform(X_fisher)

    # -------------------------- 4. CalculateFisherdiscriminant score --------------------------

    print("\nCalculatingFisherdiscriminant score...")

    fisher_scores = fisher_score_multi_class(X_fisher_scaled, y_fisher, FEATURES)

    # -------------------------- 5. Determine the positive and negative directions of features --------------------------

    print("\nDetermining the positive and negative directions of features...")

    directions = get_feature_directions(
        X_fisher_scaled, y_fisher, FEATURES, LABEL_ORDER
    )

    # -------------------------- 6. Calculate normalized weight --------------------------

    fs_values = np.array([s[1] for s in fisher_scores])

    if not np.isfinite(fs_values).all() or fs_values.sum() == 0:
        raise ValueError("The Fisher scores cannot be normalized into finite weights.")
    fs_norm = fs_values / fs_values.sum()

    weights_dict = {
        name: fs_norm[i] * directions[name] for i, (name, _) in enumerate(fisher_scores)
    }

    # -------------------------- 7. Terminal print results --------------------------

    print("\n" + "=" * 80)

    print("Features Fisher discriminant score,Positive and negative directions and final weight results:")

    print("-" * 80)

    print(
        f"{'Feature name':<25} "
        f"{'Fisher Score':<18} "
        f"{'Positive and negative':<12} "
        f"{'final weight':<12}"
    )

    print("-" * 80)

    for feat_name, score in fisher_scores:
        direction = "forward" if directions[feat_name] == 1 else "Negative"

        weight = weights_dict[feat_name]

        print(
            f"{feat_name:<25} "
            f"{score:<18.4f} "
            f"{direction:<12} "
            f"{weight:<12.4f}"
        )

    print("=" * 80)

    # -------------------------- 8. saveFisherAnalyze the results toExcel --------------------------

    print("\nSavingFisherAnalyze the results toExcel...")

    # Organize data in the order displayed on the terminal
    output_data = []

    for feat_name, score in fisher_scores:
        direction = "forward" if directions[feat_name] == 1 else "Negative"

        weight = weights_dict[feat_name]

        output_data.append(
            {
                "Feature name": feat_name,
                "Fisher Score": round(float(score), 4),
                "Positive and negative": direction,
                "final weight": round(float(weight), 4),
            }
        )

    # Convert toDataFrame
    results_df = pd.DataFrame(output_data)

    # Make sure the output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # saveExcel
    results_df.to_excel(
        args.output, index=False, engine="openpyxl", sheet_name="Fisher_Results"
    )

    print("\n" + "=" * 80)

    print("FISHER SCORE RESULTS SUCCESSFULLY SAVED")

    print("=" * 80)

    print(f"Number of samples analyzed:{len(df_fisher)}")

    print(f"Number of features:{len(FEATURES)}")

    print(f"Excelsave path:{args.output}")

    print("\nPreview of saved results:")

    print(results_df.to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    main()
