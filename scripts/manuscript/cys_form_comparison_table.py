"""Compare four CYS forms against the expert grades using Spearman correlation.

Adapted from Table_S2.py in the 2026-11-16 study scripts.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def calculate_correlations(input_path: Path, output_path: Path) -> None:
    # read Excel data
    df = pd.read_excel(input_path)

    # Confirm that necessary columns exist
    required_columns = [
        "Final_Grade",
        "CYS",
        "CYS_Additive",
        "CYS_Geometric",
        "CYS_Harmonic",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    # Filter out any rows containing null values, Ensure the accuracy of correlation calculations
    df_clean = df[required_columns].dropna()

    # Extract human final ratings as baseline (Reference)
    expert_grades = df_clean["Final_Grade"]

    # Prepare a list to store calculation results
    results = []

    # Define what needs to be calculated 4 a CYS model
    # The naming here corresponds to your target table (Table S2) in Model Column
    models = {
        "Multiplicative (CYS)": "CYS",
        "Additive": "CYS_Additive",
        "Geometric-mean": "CYS_Geometric",
        "Harmonic-mean": "CYS_Harmonic",
    }

    # Loop through each CYS Deformation and Final_Grade The Spearman correlation coefficient and P value
    for model_name, col_name in models.items():
        cys_scores = df_clean[col_name]

        # Use scipy.stats.spearmanr Calculate the correlation coefficient and Pvalue
        correlation, p_value = spearmanr(expert_grades, cys_scores)

        # Format P value: if extremely small, Use scientific notation (Such as 7.35e-56)
        # Format correlation coefficient: Reserve 4 decimal places
        results.append(
            {
                "Model": model_name,
                "Spearman r_s (Expert)": round(correlation, 4),
                "P-value": f"{p_value:.2e}",  # Convert to scientific notation format, Keep to two decimal places
            }
        )

    # Convert the result to DataFrame
    results_df = pd.DataFrame(results)

    # Print beautiful forms in the terminal, Convenient for you to check and copy directly
    print("=" * 60)
    print(
        "📊 CYS model comparison: Spearman correlation coefficient calculation results (Spearman's rank correlation)"
    )
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    # Export as Excel File
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\n✅ Results have been saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare four CYS forms against the expert grades with Spearman correlation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calculate_correlations(args.input, args.output)


if __name__ == "__main__":
    main()
