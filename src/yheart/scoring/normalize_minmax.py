"""Min-Max normalize extracted yellow-heart color traits.

Scales the ten color-space features into [0, 1] to eliminate scale differences
for downstream grading / Fisher / GMM analyses.

Usage
-----
.. code-block:: bash
    uv run minmax-normalize  # demo dataset (defaults)
    uv run minmax-normalize \\
        --input ./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/\
roi_10_color_features_with_ratio.xlsx \\
        --output ./data/normalized_data_1319.xlsx  # Figshare full dataset
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Default paths for Demo dataset
DEFAULT_INPUT_EXCEL = "./samples_images/results/roi_10_color_features_with_ratio.xlsx"
DEFAULT_OUTPUT_EXCEL = "./samples_images/results/normalized_data_1319.xlsx"


COLOR_FEATURES = [
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


def normalize_color_features(input_path: str, output_path: str):
    """Min-Max scale the ten color traits into [0, 1]."""
    if not Path(input_path).exists():
        print(f"Error: Input file not found '{input_path}'")
        return
    df = pd.read_excel(input_path)
    if missing_cols := [col for col in COLOR_FEATURES if col not in df.columns]:
        print(f"Error: Missing specified columns: {missing_cols}")
        return
    scaler = MinMaxScaler()
    df[COLOR_FEATURES] = scaler.fit_transform(df[COLOR_FEATURES])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Min-Max normalization complete. Output saved to: {output_path}")


@dataclass(frozen=True, slots=True)
class NormalizeMinmaxArgs:
    """Parsed command-line arguments."""

    input: str
    output: str


def parse_args(argv: Sequence[str] | None = None) -> NormalizeMinmaxArgs:
    p = argparse.ArgumentParser(description="Min-Max normalize yellow-heart color traits to [0,1].")
    p.add_argument(
        "--input", default=DEFAULT_INPUT_EXCEL, help="Input .xlsx of raw color features."
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_EXCEL, help="Output .xlsx of normalized features."
    )
    return cast(NormalizeMinmaxArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    normalize_color_features(args.input, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
