"""Broad-sense heritability (H²) calculation for yellow-heart traits.

Model (printed and saved to `h2_model_spec.txt` for the paper methods)::

    trait_ij = mu + G_i + e_ij,  G_i ~ N(0, σ²_G),  e_ij ~ N(0, σ²_E),

fitted per trait as a random-intercept Linear Mixed Model
(`statsmodels.MixedLM`, genotype as the random-effect grouping factor,
REML). Reported broad-sense heritability on a genotype-mean basis::

    H² = σ²_G / (σ²_G + σ²_E / n_h),

where n_h is the harmonic mean of replicate counts per genotype.

Pseudoreplication guard (Reviewer-1 Q4): multiple images of the same plant
are *technical* replicates. Pass `--replicate-col <plant_id>` to average
them to one biological observation per plant before fitting, and/or run
`--level both` to report the image-level vs. aggregated sensitivity
comparison in `variance_components.csv`. Treating every image as an
independent replicate inflates n_h and H²; the aggregated run is the
conservative reference.

Usage
-----
.. code-block:: bash
    uv run broad-heritability  # demo dataset (defaults)
    uv run broad-heritability --level both --replicate-col plant_id
"""

import argparse
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.container import BarContainer
from matplotlib.patches import Rectangle
from scipy.stats import hmean

if TYPE_CHECKING:

    @dataclass(frozen=True, slots=True)
    class MixedLMFitView:
        """Typed view of the two ``MixedLM.fit(...)`` members consumed here."""

        cov_re: pd.DataFrame
        scale: float


# Default paths for Demo dataset
DEFAULT_DATA_FILE_PATH = "./samples_images/results/CYS_1319.xlsx"
DEFAULT_OUTPUT_DIR = "./samples_images/results/Plot/"

GENOTYPE_COL = "QR"

TRAITS_MAP = {
    "S_mean": r"$\mathrm{S}$",
    "b_mean": r"$\mathrm{b^*}$",
    "ExR_mean": r"$\mathrm{ExR}$",
    "B_mean": r"$\mathrm{B}$",
    "Yellow_Ratio": "Yellow Ratio",
    "Yellow_score": "Yellow Score",
    "CYS": "CYS",
}

# Plotting parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


MODEL_SPEC = (
    "trait_ij = mu + G_i + e_ij, G_i ~ N(0, sigma2_G), e_ij ~ N(0, sigma2_E); "
    "random-intercept LMM, groups=genotype ({genotype_col}), REML via "
    "statsmodels.MixedLM; H2 = sigma2_G / (sigma2_G + sigma2_E / n_h), "
    "n_h = harmonic mean of replicate counts per genotype."
)


class H2Fit(TypedDict):
    """Variance components of one fitted trait (R1-4 methods reference)."""

    n_genotypes: int
    n_obs: int
    n_h: float
    sigma2_G: float
    sigma2_E: float
    H2: float


def resolve_trait_column(df: pd.DataFrame, col: str) -> str | None:
    """Map legacy ``1-B`` to ``B_mean``; return None when unavailable."""
    if col in df.columns:
        return col
    if col == "B_mean" and "1-B" in df.columns:
        warnings.warn(
            "Column 'B_mean' not found; using legacy '1-B' column "
            "instead. Consider renaming '1-B' to 'B_mean' upstream.",
            stacklevel=3,
        )
        return "1-B"
    return None


def fit_single_h2(df_clean: pd.DataFrame, genotype_col: str, trait_col: str) -> H2Fit | None:
    """Fit one random-intercept LMM; return variance-component dict."""
    # Quoted: pandas Series is not subscriptable at runtime.
    replicate_counts: "pd.Series[int]" = df_clean.groupby(genotype_col).size()
    k = len(replicate_counts)
    n_obs = len(df_clean)
    if k <= 1 or n_obs <= k:
        return None
    n_h = hmean(replicate_counts.to_numpy(dtype=float))
    try:
        model = smf.mixedlm(f'Q("{trait_col}") ~ 1', data=df_clean, groups=df_clean[genotype_col])
        result = cast("MixedLMFitView", model.fit(reml=True))
        sigma2_G = max(0.0, cast(float, result.cov_re.iloc[0, 0]))
        sigma2_E = result.scale
    except Exception as e:
        print(f"Failed to fit LMM model for [{trait_col}]: {e}")
        return None
    denominator = sigma2_G + (sigma2_E / n_h)
    return {
        "n_genotypes": k,
        "n_obs": n_obs,
        "n_h": round(n_h, 3),
        "sigma2_G": round(sigma2_G, 6),
        "sigma2_E": round(sigma2_E, 6),
        "H2": round(sigma2_G / denominator, 3) if denominator > 0 else 0.0,
    }


def aggregate_technical_replicates(
    df: pd.DataFrame,
    genotype_col: str,
    trait_col: str,
    replicate_col: str | None = None,
    how: str = "mean",
) -> pd.DataFrame | None:
    """Average technical (image) replicates to one row per biological unit.

    With `replicate_col` (e.g. plant ID): mean over images within each
    (genotype, replicate) cell. Without it, aggregation would leave a single
    observation per genotype (LMM unidentifiable) and returns None.
    """
    if replicate_col is None or replicate_col not in df.columns:
        return None
    sub = df[[genotype_col, replicate_col, trait_col]].dropna().copy()
    assert isinstance(sub, pd.DataFrame)
    sub[genotype_col] = sub[genotype_col].astype(str).str.strip()
    sub[replicate_col] = sub[replicate_col].astype(str).str.strip()
    grouped = sub.groupby([genotype_col, replicate_col])[trait_col]
    agg_fn: Callable[[], pd.Series] = getattr(grouped, how)
    result: pd.DataFrame = agg_fn().reset_index().rename(columns={replicate_col: "bio_replicate"})
    assert isinstance(result, pd.DataFrame)
    return result


def compute_h2_with_lmm(df: pd.DataFrame, genotype_col: str, traits_map: dict[str, str]):
    """Legacy per-trait H2 mapping (image level); prefer compute_h2_table."""
    h2_results: dict[str, float] = {}

    for col, display_name in traits_map.items():
        actual_col = resolve_trait_column(df, col)
        if actual_col is None:
            print(f"Warning: Column [{col}] not found in dataset. Skipping.")
            continue

        df_clean = df.dropna(subset=[genotype_col, actual_col]).copy()
        assert isinstance(df_clean, pd.DataFrame)
        df_clean[genotype_col] = df_clean[genotype_col].astype(str).str.strip()

        fit = fit_single_h2(df_clean, genotype_col, actual_col)
        if fit is None:
            continue
        h2_results[display_name] = fit["H2"]

    return h2_results


def compute_h2_table(
    df: pd.DataFrame,
    genotype_col: str,
    traits_map: dict[str, str],
    levels: tuple[str, ...] = ("image",),
    replicate_col: str | None = None,
    agg: str = "mean",
):
    """Full variance-component table, one row per (trait, level)."""
    rows: list[dict[str, object]] = []
    for col, display_name in traits_map.items():
        actual_col = resolve_trait_column(df, col)
        if actual_col is None:
            continue
        base = df.dropna(subset=[genotype_col, actual_col]).copy()
        assert isinstance(base, pd.DataFrame)
        base[genotype_col] = base[genotype_col].astype(str).str.strip()
        frames: dict[str, pd.DataFrame] = {}
        if "image" in levels:
            frames["image"] = base
        if "aggregate" in levels:
            agg_df = aggregate_technical_replicates(
                base, genotype_col, actual_col, replicate_col, agg
            )
            if agg_df is None:
                print(
                    "Warning: aggregate level skipped — provide "
                    "--replicate-col (e.g. plant ID); averaging all images "
                    "per accession would leave one observation per genotype."
                )
            else:
                frames["aggregate"] = agg_df
        for level, frame in frames.items():
            fit = fit_single_h2(frame, genotype_col, actual_col)
            if fit is None:
                continue
            rows.append({"trait": display_name, "level": level, **fit})
    return pd.DataFrame(
        rows,
        columns=[
            "trait",
            "level",
            "n_genotypes",
            "n_obs",
            "n_h",
            "sigma2_G",
            "sigma2_E",
            "H2",
        ],
    )


def write_model_spec(
    output_dir: str,
    genotype_col: str,
    levels: tuple[str, ...],
    replicate_col: str | None,
):
    """Write the LMM specification text quoted by the paper methods."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "h2_model_spec.txt"
    path.write_text(
        "\n".join([
            "Broad-sense heritability model (heritability.py)",
            MODEL_SPEC.format(genotype_col=genotype_col),
            f"levels: {', '.join(levels)}",
            f"genotype_col: {genotype_col}",
            f"replicate_col: {replicate_col}",
            "accession (QR) is the random effect; each row in the image level is one image.",
            "",
        ]),
        encoding="utf-8",
    )
    return str(path)


def _darken(rgb: Sequence[float], amount: float = 0.55) -> str:
    """Darken one palette entry toward black; derives edge colors from faces."""
    from matplotlib.colors import to_hex

    vals = [max(0.0, v * amount) for v in list(rgb)[:3]]
    if len(vals) != 3:
        raise ValueError(f"Palette entry must be an RGB triple, got {rgb!r}")
    return to_hex((vals[0], vals[1], vals[2]))


def plot_sci_heritability_bar(h2_results: dict[str, float], output_dir: str):
    """Render the publication H2 bar figure (SVG and PNG)."""
    if not h2_results:
        print("Error: No valid H² results to plot.")
        return

    traits = list(h2_results.keys())
    palette = sns.color_palette("tab10", n_colors=len(traits))
    face_colors = palette.as_hex()
    edge_colors = [_darken(entry) for entry in palette]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    x = np.arange(len(traits))
    width = 0.45

    bars: BarContainer = ax.bar(
        x,
        [h2_results[t] for t in traits],
        width=width,
        color=face_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )

    for patch, trait in zip(cast(Sequence[Rectangle], bars), traits):
        bar = patch
        val = h2_results[trait]
        cx: float = bar.get_x() + bar.get_width() / 2.0
        top: float = bar.get_height() + 0.006
        ax.text(
            cx,
            top,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#1E293B",
        )

    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(r"Broad-Sense Heritability ($H^2$)", fontsize=14, fontweight="bold", labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=12, rotation=28, ha="right")

    ax.tick_params(axis="y", labelsize=11.5, length=4, width=1.0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#CBD5E1", zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")

    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "Figure_S1_Broad_Sense_Heritability_LMM.svg"
    png_path = out_dir / "Figure_S1_Broad_Sense_Heritability_LMM.png"

    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Heritability analysis complete. Figure saved to: {output_dir}")


@dataclass(frozen=True, slots=True)
class HeritabilityArgs:
    """Parsed command-line arguments."""

    data: str
    output: str
    level: str
    replicate_col: str | None
    agg: str


def parse_args(argv: Sequence[str] | None = None) -> HeritabilityArgs:
    p = argparse.ArgumentParser(
        description="Broad-sense heritability (H2) via LMM for yellow-heart traits."
    )
    p.add_argument(
        "--data",
        default=DEFAULT_DATA_FILE_PATH,
        help="Input .xlsx with genotype (" + GENOTYPE_COL + ") and trait columns.",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder for the H2 figure and tables."
    )
    p.add_argument(
        "--level",
        choices=["image", "aggregate", "both"],
        default="image",
        help="image: every image is a replicate (legacy); "
        "aggregate: average technical replicates per "
        "--replicate-col first; both: sensitivity comparison.",
    )
    p.add_argument(
        "--replicate-col",
        default=None,
        help="Biological-replicate ID column (e.g. plant_id) used by the aggregate level.",
    )
    p.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation for technical replicates.",
    )
    return cast(HeritabilityArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    if not Path(args.data).exists():
        print(f"Error: Dataset file not found at '{args.data}'")
        return
    df_raw = pd.read_excel(args.data)
    if "1-B" in df_raw.columns and "B_mean" not in df_raw.columns:
        df_raw["B_mean"] = df_raw["1-B"]

    levels = ("image", "aggregate") if args.level == "both" else (args.level,)
    table = compute_h2_table(
        df_raw,
        GENOTYPE_COL,
        TRAITS_MAP,
        levels=levels,
        replicate_col=args.replicate_col,
        agg=args.agg,
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not table.empty:
        csv_path = out_dir / "variance_components.csv"
        table.to_csv(csv_path, index=False)
        print(table.to_string(index=False))
        print(f"Variance components saved to: {csv_path}")
    spec_path = write_model_spec(args.output, GENOTYPE_COL, levels, args.replicate_col)
    print(f"Model spec saved to: {spec_path}")

    img = table if table.empty else table[table["level"] == "image"]
    h2_dict: dict[str, float] = {} if table.empty else dict(zip(img["trait"], img["H2"]))
    if not h2_dict and not table.empty:
        h2_dict = dict(zip(table["trait"], table["H2"]))
    plot_sci_heritability_bar(h2_dict, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
