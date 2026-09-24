r"""GMM-based color trait modeling and grade-threshold derivation.

Fits a Gaussian Mixture Model to the Comprehensive Yellow Score (CYS) /
``Yellow_score`` distribution, selects the number of components by BIC, derives
decision thresholds at component-intersection points, exports the graded
dataset, and renders a multi-panel validation figure (BIC curve + component
densities and thresholds).

Usage
-----
.. code-block:: bash
    uv run gmm-cys  # demo dataset (defaults)
    uv run gmm-cys \
        --data ./data/CYS_1319.xlsx \
        --output ./data/Plot  # Figshare full dataset
"""

import argparse
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Runtime truth (scipy rv_continuous.pdf docstring: "Returns pdf : ndarray"; the
# concrete case here is always float64 — verified) that we state ourselves
# because scipy-stubs for `norm` resolve to `Unknown` in both checkers.
PdfArray = npt.NDArray[np.float64]
if TYPE_CHECKING:
    from matplotlib.axes import Axes


def norm_pdf(x: npt.NDArray[np.float64], loc: float, scale: float) -> PdfArray:
    """Thin wrapper pinning the runtime return type of ``norm.pdf``."""
    return np.asarray(norm.pdf(x, loc, scale), dtype=np.float64)


# Default paths for Demo dataset
DEFAULT_DATA_FILE_PATH = "./samples_images/results/CYS_1319.xlsx"
DEFAULT_OUTPUT_DIR = "./samples_images/results/Plot"


FORCE_COMPONENTS = None
MAX_COMPONENTS = 8
RANDOM_STATE = 42
N_INIT = 15


# Plotting parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def _fit_gmm_models(
    X: npt.NDArray[np.float64],
    max_components: int,
    random_state: int,
    n_init: int,
) -> tuple[list[float], list[GaussianMixture]]:
    """Fit candidate GMMs and return their BIC scores."""
    bics: list[float] = []
    models: list[GaussianMixture] = []
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=n_init)
        gmm.fit(X)
        bics.append(cast(float, gmm.bic(X)))
        models.append(gmm)
    return bics, models


def _ordered_gmm_parameters(
    model: GaussianMixture,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return fitted GMM weights, means, and standard deviations by mean."""
    assert model.weights_ is not None
    assert model.means_ is not None
    assert model.covariances_ is not None

    sorted_indices = np.argsort(cast(np.ndarray, model.means_).flatten())
    return (
        cast(np.ndarray, model.weights_)[sorted_indices],
        cast(np.ndarray, model.means_).flatten()[sorted_indices],
        np.sqrt(cast(np.ndarray, model.covariances_).flatten())[sorted_indices],
    )


def _find_thresholds(
    weights: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    std_devs: npt.NDArray[np.float64],
    x_grid: PdfArray,
) -> list[float]:
    """Find decision thresholds at adjacent GMM component intersections."""
    thresholds: list[float] = []
    for i in range(len(means) - 1):
        pdf1: PdfArray = weights[i] * norm_pdf(x_grid, means[i], std_devs[i])
        pdf2: PdfArray = weights[i + 1] * norm_pdf(x_grid, means[i + 1], std_devs[i + 1])
        crossings = np.where(np.diff(np.sign(pdf1 - pdf2)))[0]
        if len(crossings) > 0:
            midpoint = (means[i] + means[i + 1]) / 2.0
            closest_idx = int(crossings[np.argmin(np.abs(x_grid[crossings] - midpoint))])
            threshold = float(x_grid[closest_idx])
        else:
            threshold = (means[i] + means[i + 1]) / 2.0
        thresholds.append(threshold)
    return thresholds


def _load_and_prepare(file_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        return None
    if "Yellow_score" not in df.columns:
        print("Error: 'Yellow_score' column not found in dataset.")
        return None
    df_clean = df.dropna(subset=["Yellow_score"]).copy()
    assert isinstance(df_clean, pd.DataFrame)
    df_clean["Yellow_score"] = df_clean["Yellow_score"].clip(lower=0)
    return df_clean


def _export_graded_dataset(
    df_clean: pd.DataFrame,
    best_k: int,
    thresholds: list[float],
    file_path: str,
    output_dir: str,
) -> None:
    bins = [-np.inf] + thresholds + [np.inf]
    labels = range(1, best_k + 1)
    df_export = df_clean.copy()
    df_export["Scientific_Grade"] = pd.cut(df_export["Yellow_score"], bins=bins, labels=labels)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    excel_name = Path(file_path).name.replace(".xlsx", f"_GMM_Final_{best_k}Classes.xlsx")
    output_excel = Path(output_dir) / excel_name
    df_export.to_excel(output_excel, index=False)


def _plot_gmm_validation(
    bics: list[float],
    best_k: int,
    max_components: int,
    df_clean: pd.DataFrame,
    weights: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    std_devs: npt.NDArray[np.float64],
    thresholds: list[float],
    output_dir: str,
):
    axes: "Sequence[Axes]"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    ax1.text(-0.12, 1.05, "(A)", transform=ax1.transAxes, fontsize=18, fontweight="bold", va="top")
    ax1.plot(
        range(1, max_components + 1),
        bics,
        marker="o",
        linestyle="-",
        color="#2C7BB6",
        linewidth=2.5,
        markersize=8,
    )
    ax1.plot(best_k, bics[best_k - 1], marker="o", color="#D7191C", markersize=12, zorder=5)
    ax1.axvline(
        x=best_k, color="#D7191C", linestyle="--", linewidth=1.8, label=f"Optimal K = {best_k}"
    )
    ax1.set_title("Bayesian Information Criterion (BIC)", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Number of Gaussian Components (K)", fontsize=14)
    ax1.set_ylabel("BIC Score", fontsize=14)
    ax1.set_xticks(range(1, max_components + 1))
    ax1.set_xlim(0.5, max_components + 0.5)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=12, frameon=False)
    sns.despine(ax=ax1, top=True, right=True)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)

    ax2 = axes[1]
    ax2.text(-0.12, 1.05, "(B)", transform=ax2.transAxes, fontsize=18, fontweight="bold", va="top")
    sns.histplot(
        data=df_clean,
        x="Yellow_score",
        bins=65,
        stat="density",
        color="#D3D3D3",
        edgecolor="white",
        alpha=0.8,
        ax=ax2,
        label="Empirical Distribution",
    )
    base_colors = ["#8BA888", "#F2DD72", "#F2B705", "#F27405", "#BF212E", "#9467bd", "#8c564b"]
    colors = base_colors[:best_k]
    x_max = df_clean["Yellow_score"].max() * 1.05
    x_grid: PdfArray = np.linspace(0, x_max, 20000, dtype=np.float64)
    for i in range(best_k):
        pdf: PdfArray = weights[i] * norm_pdf(x_grid, means[i], std_devs[i])
        ax2.plot(
            x_grid, pdf, color=colors[i], linewidth=3, alpha=0.9, label=f"Subpopulation {i + 1}"
        )
        ax2.fill_between(x_grid, pdf, alpha=0.2, color=colors[i])
    current_y_max = ax2.get_ylim()[1]
    ax2.set_ylim(0, current_y_max * 1.15)
    y_max = ax2.get_ylim()[1]
    for i, t in enumerate(thresholds):
        ax2.axvline(x=t, ymin=0, ymax=1, color="#404040", linestyle=":", linewidth=2)
        ax2.text(
            t + 0.008,
            y_max * 0.88,
            r"$T_{" + str(i + 1) + r"} = " + f"{t:.3f}$",
            color="#222222",
            fontweight="bold",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2},
        )
    ax2.set_title("GMM Subpopulations & Derived Thresholds", fontsize=16, fontweight="bold", pad=15)
    ax2.set_xlabel("Comprehensive Yellow Score (CYS)", fontsize=14)
    ax2.set_ylabel("Probability Density", fontsize=14)
    ax2.legend(fontsize=11, loc="upper right", frameon=False)
    ax2.set_xlim(left=0, right=x_max)
    ax2.tick_params(axis="both", which="major", labelsize=12)
    sns.despine(ax=ax2, top=True, right=True)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    out_dir = Path(output_dir)
    out_svg = out_dir / "GMM_Final_Analysis.svg"
    out_png = out_dir / "GMM_Final_Analysis.png"
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.savefig(out_png, format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def find_optimal_gmm_and_thresholds(
    file_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    force_components: int | None = FORCE_COMPONENTS,
    max_components: int = MAX_COMPONENTS,
    random_state: int = RANDOM_STATE,
    n_init: int = N_INIT,
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*|.*did not converge.*")
        df_clean = _load_and_prepare(file_path)
        if df_clean is None:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        X = df_clean["Yellow_score"].to_numpy(dtype=float).reshape(-1, 1)
        bics, models = _fit_gmm_models(X, max_components, random_state, n_init)
        optimal_k = int(np.argmin(bics)) + 1
        best_k = force_components if force_components is not None else optimal_k
        best_gmm = models[best_k - 1]
        weights, means, std_devs = _ordered_gmm_parameters(best_gmm)
        x_max = X.max() * 1.05
        x_grid: PdfArray = np.linspace(0, x_max, 20000, dtype=np.float64)
        thresholds = _find_thresholds(weights, means, std_devs, x_grid)
        _export_graded_dataset(df_clean, best_k, thresholds, file_path, output_dir)
        _plot_gmm_validation(
            bics, best_k, max_components, df_clean, weights, means, std_devs, thresholds, output_dir
        )
        print(f"GMM analysis complete. Output saved to: {output_dir}")


@dataclass(frozen=True, slots=True)
class GmmGradingArgs:
    """Parsed command-line arguments."""

    data: str
    output: str
    force_components: int | None
    max_components: int
    random_state: int


def parse_args(argv: Sequence[str] | None = None) -> GmmGradingArgs:
    p = argparse.ArgumentParser(
        description="GMM-based CYS modeling and grade-threshold derivation."
    )
    p.add_argument(
        "--data", default=DEFAULT_DATA_FILE_PATH, help="Input .xlsx with a 'Yellow_score' column."
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR, help="Output folder for plots and graded Excel."
    )
    p.add_argument(
        "--force-components",
        type=int,
        default=FORCE_COMPONENTS,
        help="Force a specific GMM component count (default: BIC-optimal).",
    )
    p.add_argument(
        "--max-components",
        type=int,
        default=MAX_COMPONENTS,
        help="Maximum number of BIC-searched components.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random state for GMM reproducibility.",
    )
    return cast(GmmGradingArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    find_optimal_gmm_and_thresholds(
        args.data, args.output, args.force_components, args.max_components, args.random_state
    )


if __name__ == "__main__":
    main(sys.argv[1:])
