"""Append additive, geometric-mean and harmonic-mean CYS columns.

Adapted from 4_CYS_Add_Geo_Har.py in the 2026-11-16 study scripts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_alternative_cys(file_path: Path, output_path: Path) -> None:
    print("\n" + "=" * 65)
    print("       HPJ | ALTERNATIVE CYS MODEL CALCULATION")
    print("=" * 65)

    print(f"\nReading file:{file_path.name}")

    df = pd.read_excel(file_path)
    df.columns = df.columns.astype(str).str.strip()

    # =====================================================
    # 3. Confirm data column
    # =====================================================

    YR_COL = "Yellow_Ratio"  # BColumn:YR
    YS_COL = "Yellow_score"  # LColumn:YS
    CYS_COL = "CYS"  # MColumn:original multiplicative model

    required_cols = [YR_COL, YS_COL, CYS_COL]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"ExcelRequired column is missing in:{missing}")

    # =====================================================
    # 4. Extract YR,YS and original CYS
    # =====================================================

    yr = pd.to_numeric(df[YR_COL], errors="coerce")
    ys = pd.to_numeric(df[YS_COL], errors="coerce")
    cys_original = pd.to_numeric(df[CYS_COL], errors="coerce")

    valid = yr.notna() & ys.notna()

    if ((yr[valid] < 0) | (ys[valid] < 0)).any():
        raise ValueError("YRorYSThere is a negative value,Please check the original data first.")

    print(f"\nTotal number of samples:{len(df)}")
    print(f"Efficiently calculate the number of samples:{valid.sum()}")
    print(f"Invalid or missing sample number:{(~valid).sum()}")

    # =====================================================
    # 5. Calculate three alternatives CYS model
    # =====================================================

    # (1) Additive model
    # CYS_Additive = 0.5 * (YS + YR)

    cys_additive = 0.5 * (ys + yr)
    cys_additive = cys_additive.where(valid)

    # (2) Geometric-mean model
    # CYS_Geometric = sqrt(YS * YR)

    cys_geometric = np.sqrt((ys * yr).clip(lower=0))
    cys_geometric = cys_geometric.where(valid)

    # (3) Harmonic-mean model
    # CYS_Harmonic = 2 * YS * YR / (YS + YR)

    denominator = ys + yr

    cys_harmonic = pd.Series(np.nan, index=df.index, dtype=float)

    nonzero = valid & (denominator > 0)

    cys_harmonic.loc[nonzero] = 2 * ys[nonzero] * yr[nonzero] / denominator[nonzero]

    # when YS = 0 And YR = 0 time,Define the harmonic mean as0
    both_zero = valid & (denominator == 0)
    cys_harmonic.loc[both_zero] = 0.0

    # =====================================================
    # 6. Insert the three model results into the original CYS behind
    # =====================================================

    # Delete old calculation results that may already exist,avoid duplicate columns
    alternative_cols = ["CYS_Additive", "CYS_Geometric", "CYS_Harmonic"]

    df.drop(
        columns=[col for col in alternative_cols if col in df.columns], inplace=True
    )

    # Locate originalCYSColumn
    insert_position = df.columns.get_loc(CYS_COL) + 1

    # follow N,O,P sequential insertion
    df.insert(insert_position, "CYS_Additive", cys_additive)

    df.insert(insert_position + 1, "CYS_Geometric", cys_geometric)

    df.insert(insert_position + 2, "CYS_Harmonic", cys_harmonic)

    # =====================================================
    # 7. Check original CYS Is it equal to YS × YR
    # =====================================================

    check_valid = valid & cys_original.notna()

    if check_valid.any():
        cys_check = ys * yr

        differences = (cys_original[check_valid] - cys_check[check_valid]).abs()

        max_error = differences.max()

        print("\noriginal CYS Consistency check:")
        print(f"maximum absolute error:{max_error:.12f}")

        if np.allclose(
            cys_original[check_valid], cys_check[check_valid], rtol=1e-7, atol=1e-10
        ):
            print("result:original CYS with YS × YR consistent.")
        else:
            print("warning:Partially original CYS with YS × YR inconsistent.")
            print("original CYS Reserved,not covered.")

    # =====================================================
    # 8. save Excel
    # =====================================================

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")

    # =====================================================
    # 9. Terminal output results
    # =====================================================

    print("\n" + "=" * 65)
    print("          ALTERNATIVE CYS CALCULATION RESULTS")
    print("=" * 65)

    print("\nAdd new calculated column:")
    for column in alternative_cols:
        print(column)

    print("\nCalculation formula:")
    print("Multiplicative : YS × YR")
    print("Additive       : 0.5 × (YS + YR)")
    print("Geometric      : sqrt(YS × YR)")
    print("Harmonic       : 2 × YS × YR / (YS + YR)")

    print("\nbefore5Row result preview:")

    preview_cols = [
        "filename",
        YR_COL,
        YS_COL,
        CYS_COL,
        "CYS_Additive",
        "CYS_Geometric",
        "CYS_Harmonic",
    ]

    existing_cols = [col for col in preview_cols if col in df.columns]

    print(df[existing_cols].head().to_string(index=False))

    print("\n" + "=" * 65)
    print("CALCULATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"Total number of samples:{len(df)}")
    print(f"Efficiently calculate the number of samples:{valid.sum()}")
    print(f"output file:{output_path}")

    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append the three alternative forms of CYS to an input table."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calculate_alternative_cys(args.input, args.output)


if __name__ == "__main__":
    main()
