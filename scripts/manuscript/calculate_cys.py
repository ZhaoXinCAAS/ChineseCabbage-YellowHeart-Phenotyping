"""Recompute the full-precision YS and CYS formulas from the study workbook.

The variable names and four-decimal coefficients come from the archived
1319-image CYS.xlsx workbook. The output preserves its column order.
"""

import argparse
from pathlib import Path

import pandas as pd


def calculate_cys(normalized: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "filename",
        "Yellow_Ratio",
        "B_mean_norm",
        "S_mean_norm",
        "b_mean_norm",
        "ExR_mean_norm",
    ]
    missing = set(columns) - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    result = normalized[columns].copy()
    for name in columns[1:]:
        result[name] = pd.to_numeric(result[name], errors="raise")

    result["1-Bnorm"] = 1 - result["B_mean_norm"]
    result["ExR_weight"] = 0.2931
    result["S_weight"] = 0.2595
    result["b_weight"] = 0.2272
    result["B_weight"] = 0.2202
    result["Yellow_score"] = (
        result["ExR_weight"] * result["ExR_mean_norm"]
        + result["S_weight"] * result["S_mean_norm"]
        + result["b_weight"] * result["b_mean_norm"]
        + result["B_weight"] * result["1-Bnorm"]
    )
    result["CYS"] = result["Yellow_score"] * result["Yellow_Ratio"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, required=True, help="Normalized feature workbook"
    )
    parser.add_argument("--output", type=Path, required=True, help="CYS workbook")
    args = parser.parse_args()

    result = calculate_cys(pd.read_excel(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(args.output, index=False, engine="openpyxl")
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
