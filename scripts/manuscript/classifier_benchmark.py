"""Benchmark five classifiers with stratified row-level cross-validation.

Adapted from 3_2_FDA_SVMTable_S1.py in the 2026-11-16 study scripts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
)  # If there is a strain series, it can be changed to StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the five-classifier benchmark from the study."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Reading data: {args.input} ..")
    df = pd.read_excel(args.input)

    # 2. Extract features and labels
    feature_cols = [
        "R_mean",
        "G_mean",
        "B_mean",
        "H_circular_mean_deg",
        "S_mean",
        "V_mean",
        "L_mean",
        "a_mean",
        "b_mean",
        "ExR_mean",
    ]

    X = df[feature_cols].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    # 3. Initialize comparison model
    models = {
        "FDA (LDA)": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(
            penalty="l2", max_iter=1000, random_state=42
        ),
        "SVM (Linear)": SVC(kernel="linear", random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # 4. settings 5 fold-stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {name: {"acc": [], "f1": []} for name in models.keys()}

    print("Running five-fold stratified cross-validation with fold-local scaling..")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # [critical fix]: Fit only on training set Scaler, It is strictly prohibited to cross data into the test set!
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name]["acc"].append(accuracy_score(y_test, y_pred))
            results[name]["f1"].append(f1_score(y_test, y_pred, average="macro"))

    # 5. Organize the results and output
    output_data = []
    print("\n" + "=" * 75)
    print(
        f"{'Model':<20} | {'Dim':<5} | {'Accuracy (Mean±SD)':<22} | {'Macro-F1 (Mean±SD)'}"
    )
    print("-" * 75)

    for name in models.keys():
        mean_acc = np.mean(results[name]["acc"]) * 100
        std_acc = np.std(results[name]["acc"]) * 100
        mean_f1 = np.mean(results[name]["f1"]) * 100
        std_f1 = np.std(results[name]["f1"]) * 100

        acc_str = f"{mean_acc:.2f} ± {std_acc:.2f}%"
        f1_str = f"{mean_f1:.2f} ± {std_f1:.2f}%"

        print(f"{name:<20} | {'10':<5} | {acc_str:<22} | {f1_str}")

        output_data.append(
            {
                "Model": name,
                "Input Feature Dim": 10,
                "Accuracy (Mean±SD)": acc_str,
                "Macro-F1 (Mean±SD)": f1_str,
            }
        )

    print("=" * 75)

    output_df = pd.DataFrame(output_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(args.output, index=False)
    print(f"\n✅ Results were successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
