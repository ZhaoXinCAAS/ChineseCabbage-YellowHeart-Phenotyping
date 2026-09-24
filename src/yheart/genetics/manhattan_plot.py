"""Manhattan plot visualization for GWAS results."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandera.errors import SchemaError

from yheart.genetics.schemas import gwas_col, validate_gwas_table


@dataclass(frozen=True, slots=True)
class ManhattanArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    chr_label: str
    sig_threshold: float
    suggestive_threshold: float
    title: str


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Generate Manhattan plot from GWAS results.")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="GWAS results CSV (must contain SNP, CHR, BP, P).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./manhattan_plot.png",
        help="Output plot file path.",
    )
    p.add_argument(
        "--chr-label",
        type=str,
        default="Chromosome",
        help="X-axis label.",
    )
    p.add_argument(
        "--sig-threshold",
        type=float,
        default=5e-8,
        help="Significance threshold line.",
    )
    p.add_argument(
        "--suggestive-threshold",
        type=float,
        default=1e-5,
        help="Suggestive threshold; set 0 to disable.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Manhattan Plot",
        help="Plot title.",
    )
    return cast(ManhattanArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    if not args.input:
        print("Error: --input is required.")
        sys.exit(1)

    df = pd.read_csv(args.input, dtype={"SNP": str, "CHR": str})

    if df.empty:
        print("Error: Input CSV is empty.")
        sys.exit(1)

    try:
        gdf = validate_gwas_table(df)
    except SchemaError as e:
        print(f"Error: GWAS table failed validation: {e}")
        sys.exit(1)
    gdf["P"] = gwas_col(gdf, "P").replace(0, 1e-300)
    gdf = gdf[gwas_col(gdf, "P") > 0].copy()
    if gdf.empty:
        print("Error: No valid P values (all zero or non-positive).")
        sys.exit(1)

    gdf["-log10P"] = -np.log10(gwas_col(gdf, "P"))

    chr_col = gwas_col(gdf, "CHR")
    chrom_df = chr_col.value_counts().sort_index().reset_index()
    chrom_df.columns = ["CHR", "count"]
    chrom_df["offset"] = chrom_df["count"].cumsum() - chrom_df["count"]
    offset_map: dict[str | int | float, int] = dict(
        zip(chrom_df["CHR"].tolist(), chrom_df["offset"].tolist(), strict=False)
    )
    chrom_df["center"] = chrom_df["offset"] + chrom_df["count"] / 2
    chrom_colors = sns.color_palette("tab10", n_colors=len(chrom_df)).as_hex()
    color_map = {
        chr_: chrom_colors[i % len(chrom_colors)] for i, chr_ in enumerate(chrom_df["CHR"])
    }

    bp = gwas_col(gdf, "BP").to_numpy()
    mapped = chr_col.map(offset_map).fillna(0).to_numpy()
    gdf["x"] = bp + mapped

    for chr_ in chrom_df["CHR"]:
        sub = cast(pd.DataFrame, gdf[gwas_col(gdf, "CHR") == chr_])
        plt.scatter(
            sub["x"],
            sub["-log10P"],
            s=20,
            c=[color_map[chr_]],
            label=str(chr_),
        )
    if args.suggestive_threshold > 0:
        plt.axhline(
            y=-np.log10(args.suggestive_threshold),
            color="blue",
            linestyle="--",
            linewidth=1,
            label=f"Suggestive (P={args.suggestive_threshold})",
        )
    plt.axhline(
        y=-np.log10(args.sig_threshold),
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Significant (P={args.sig_threshold})",
    )

    top5 = gdf.nsmallest(5, columns="P")
    for _, row in top5.iterrows():
        plt.annotate(
            str(row["SNP"]),
            xy=(row["x"], row["-log10P"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.8,
        )

    plt.xlabel(args.chr_label)
    plt.ylabel("-log10(P)")
    plt.title(args.title)
    plt.tight_layout()

    plt.xticks(
        chrom_df["center"],
        labels=[str(c) for c in chrom_df["CHR"]],
        rotation=45,
        ha="right",
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    legend_handles = [h for h, lbl in zip(handles, labels, strict=False) if "P=" in lbl]
    legend_labels = [lbl for lbl in labels if "P=" in lbl]
    plt.legend(legend_handles, legend_labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Manhattan plot saved to: {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
