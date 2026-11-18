"""Fit the CYS Gaussian mixture and generate the BIC, density and violin panels.

Adapted from 3_GMM_CYS.py in the 2026-11-16 study scripts.
"""

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture


# ============================================================
# 1. ANALYSIS SETTINGS
# ============================================================

TARGET_COL = "CYS"
MAX_COMPONENTS = 8

RANDOM_STATE = 42
N_INIT = 20
REG_COVAR = 1e-6

# Figure dimensions:
# The reference composite figure is about 17 cm wide in total.
# This script generates ONLY the left A-C column, so use about 8.3 cm.
FIG_WIDTH_CM = 8.3
FIG_HEIGHT_CM = 17.0

# Typography and output
FONT_FAMILY = "Arial"
FONT_SIZE_PT = 7.5
SAVE_PNG = True
PNG_DPI = 600

# CYS is theoretically expected to be between 0 and 1.
# Values outside this interval are reported as an error rather than silently clipped.
EXPECTED_CYS_MIN = 0.0
EXPECTED_CYS_MAX = 1.0

# Five-grade names and colors, ordered from low to high CYS.
FIVE_GRADE_NAMES = [
    "White",
    "Light Yellow",
    "Yellow",
    "Deep Yellow",
    "Extreme Yellow",
]

GRADE_COLORS = [
    "#D9D5C8",  # White
    "#E9DFA8",  # Light Yellow
    "#F2C85B",  # Yellow
    "#E99A4B",  # Deep Yellow
    "#D96B35",  # Extreme Yellow
]

# ============================================================
# 2. GLOBAL PLOT SETTINGS
# ============================================================

CM_TO_INCH = 1.0 / 2.54

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "axes.unicode_minus": False,
        # Keep SVG text editable instead of converting text to vector paths.
        "svg.fonttype": "none",
        # Keep PDF/PostScript fonts as TrueType fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# ============================================================
# 3. DATA READING AND VALIDATION
# ============================================================


def read_input_table(file_path):
    """Read Excel or CSV input."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{file_path}\n" "Please check the --input path."
        )

    suffix = file_path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Input file must be .xlsx, .xls, or .csv")

    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Column '{TARGET_COL}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_cys_data(df):
    """Convert CYS to numeric and remove missing/non-finite observations."""
    df_clean = df.copy()
    df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL], errors="coerce")

    valid_mask = df_clean[TARGET_COL].notna() & np.isfinite(
        df_clean[TARGET_COL].to_numpy(dtype=float)
    )

    n_original = len(df_clean)
    df_clean = df_clean.loc[valid_mask].copy()

    values = df_clean[TARGET_COL].to_numpy(dtype=float)

    if len(values) < 10:
        raise ValueError(
            f"Only {len(values)} valid CYS observations were found. "
            "At least 10 observations are required for this analysis."
        )

    if np.any(values < EXPECTED_CYS_MIN - 1e-8) or np.any(
        values > EXPECTED_CYS_MAX + 1e-8
    ):
        raise ValueError(
            "CYS values outside the expected [0, 1] range were detected.\n"
            f"Observed minimum = {values.min():.6f}; "
            f"maximum = {values.max():.6f}.\n"
            "Please check the CYS calculation and input column. "
            "The script will not silently clip these values."
        )

    n_removed = n_original - len(df_clean)
    print(f"Original rows: {n_original}")
    print(f"Valid CYS rows: {len(df_clean)}")
    print(f"Rows excluded because CYS was missing/non-numeric: {n_removed}")
    print(f"CYS range: {values.min():.6f} to {values.max():.6f}")

    return df_clean, values


# ============================================================
# 4. FIT GMM MODELS AND SELECT K USING BIC
# ============================================================


def fit_gmm_models(values, force_components: int | None = None):
    """Fit GMM models from K=1 to MAX_COMPONENTS and select the lowest BIC."""
    max_k = min(MAX_COMPONENTS, len(values) - 1)

    if max_k < 1:
        raise ValueError("Not enough observations for GMM fitting.")

    X = values.reshape(-1, 1)
    models = []
    bic_scores = []

    print("\nFitting GMM models and calculating BIC...")

    for k in range(1, max_k + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            reg_covar=REG_COVAR,
        )
        model.fit(X)
        bic = model.bic(X)

        models.append(model)
        bic_scores.append(float(bic))

        print(f"K = {k}: BIC = {bic:.4f}")

    automatic_k = int(np.argmin(bic_scores) + 1)

    if force_components is None:
        selected_k = automatic_k
        selection_label = f"Optimal K = {selected_k}"
    else:
        if force_components < 1 or force_components > max_k:
            raise ValueError(f"--components must be between 1 and {max_k}.")
        selected_k = int(force_components)
        selection_label = f"Forced K = {selected_k}"

    print(f"\nBIC-selected K: {automatic_k}")
    print(f"Model used for grading: K = {selected_k}")

    selected_model = models[selected_k - 1]

    bic_table = pd.DataFrame(
        {
            "K": np.arange(1, max_k + 1),
            "BIC": bic_scores,
            "BIC_selected": [k == automatic_k for k in range(1, max_k + 1)],
            "Model_used_for_grading": [k == selected_k for k in range(1, max_k + 1)],
        }
    )

    return (
        models,
        bic_scores,
        automatic_k,
        selected_k,
        selection_label,
        selected_model,
        bic_table,
    )


# ============================================================
# 5. SORT GMM COMPONENTS AND CALCULATE DENSITY INTERSECTIONS
# ============================================================


def get_sorted_gmm_parameters(model):
    """Return component weights, means and SDs ordered by increasing mean."""
    weights = model.weights_.astype(float)
    means = model.means_.reshape(-1).astype(float)

    covariances = model.covariances_.reshape(model.n_components, -1)
    variances = covariances[:, 0].astype(float)
    std_devs = np.sqrt(variances)

    order = np.argsort(means)

    return (
        weights[order],
        means[order],
        std_devs[order],
    )


def weighted_density_difference(x, w1, mu1, sd1, w2, mu2, sd2):
    """Difference between two weighted Gaussian probability densities."""
    return w1 * norm.pdf(x, loc=mu1, scale=sd1) - w2 * norm.pdf(x, loc=mu2, scale=sd2)


def find_adjacent_intersection(w1, mu1, sd1, w2, mu2, sd2):
    """
    Find the weighted-density intersection between adjacent components.

    The preferred threshold is a root between the two component means.
    If no such root exists, the midpoint of the means is used as a fallback
    and is explicitly flagged in the output.
    """
    lower = min(mu1, mu2)
    upper = max(mu1, mu2)

    grid = np.linspace(lower, upper, 5001)

    diff = weighted_density_difference(grid, w1, mu1, sd1, w2, mu2, sd2)

    roots = []

    for i in range(len(grid) - 1):
        y1 = diff[i]
        y2 = diff[i + 1]

        if y1 == 0:
            roots.append(float(grid[i]))
        elif y1 * y2 < 0:
            root = brentq(
                weighted_density_difference,
                grid[i],
                grid[i + 1],
                args=(w1, mu1, sd1, w2, mu2, sd2),
            )
            roots.append(float(root))

    if roots:
        midpoint = (mu1 + mu2) / 2.0
        threshold = min(roots, key=lambda x: abs(x - midpoint))
        method = "Weighted-density intersection"
        fallback_used = False
    else:
        threshold = (mu1 + mu2) / 2.0
        method = "Midpoint fallback: no intersection between means"
        fallback_used = True

    return threshold, method, fallback_used


def calculate_thresholds(weights, means, std_devs):
    """Calculate thresholds between adjacent sorted Gaussian components."""
    thresholds = []
    records = []

    for i in range(len(means) - 1):
        threshold, method, fallback = find_adjacent_intersection(
            weights[i],
            means[i],
            std_devs[i],
            weights[i + 1],
            means[i + 1],
            std_devs[i + 1],
        )

        thresholds.append(threshold)

        records.append(
            {
                "Threshold": f"T{i + 1}",
                "Value": threshold,
                "Lower_component_mean": means[i],
                "Upper_component_mean": means[i + 1],
                "Calculation_method": method,
                "Midpoint_fallback_used": fallback,
            }
        )

        print(f"{'T' + str(i + 1)} = {threshold:.6f} " f"({method})")

    thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) > 1 and np.any(np.diff(thresholds) <= 0):
        raise RuntimeError(
            "Calculated thresholds are not strictly increasing. "
            "Please inspect the fitted GMM components."
        )

    return thresholds, pd.DataFrame(records)


# ============================================================
# 6. ASSIGN GMM GRADES AND CALCULATE GRADE SUMMARIES
# ============================================================


def assign_grades(df_clean, values, thresholds, selected_k):
    """Assign ordered grades using the calculated thresholds."""
    bins = [-np.inf] + thresholds.tolist() + [np.inf]
    grade_codes = list(range(1, selected_k + 1))

    grade_series = pd.cut(
        values,
        bins=bins,
        labels=grade_codes,
        include_lowest=True,
        right=True,
        ordered=True,
    )

    result = df_clean.copy()
    result["CYS_GMM_Grade"] = grade_series.astype("Int64")

    if selected_k == 5:
        name_map = {i + 1: FIVE_GRADE_NAMES[i] for i in range(5)}
        result["CYS_GMM_Grade_Name"] = result["CYS_GMM_Grade"].map(name_map)
        grade_names = FIVE_GRADE_NAMES.copy()
    else:
        name_map = {i + 1: f"Grade {i + 1}" for i in range(selected_k)}
        result["CYS_GMM_Grade_Name"] = result["CYS_GMM_Grade"].map(name_map)
        grade_names = [f"Grade {i + 1}" for i in range(selected_k)]

        warnings.warn(
            f"The selected K is {selected_k}, not 5. "
            "Generic Grade labels are used; do not describe these as "
            "the five original yellow-heart categories without justification."
        )

    summary_records = []

    for grade in grade_codes:
        group = result.loc[result["CYS_GMM_Grade"] == grade, TARGET_COL].astype(float)

        summary_records.append(
            {
                "Grade": grade,
                "Grade_name": grade_names[grade - 1],
                "N": int(group.size),
                "Mean_CYS": float(group.mean()) if group.size else np.nan,
                "SD_CYS": float(group.std(ddof=1)) if group.size > 1 else np.nan,
                "Median_CYS": float(group.median()) if group.size else np.nan,
                "Min_CYS": float(group.min()) if group.size else np.nan,
                "Max_CYS": float(group.max()) if group.size else np.nan,
            }
        )

    summary = pd.DataFrame(summary_records)

    if int(summary["N"].sum()) != len(result):
        raise RuntimeError(
            "Some CYS observations were not assigned to a grade. "
            "Please inspect the thresholds."
        )

    return result, summary, grade_names


# ============================================================
# 7. PLOT PANEL A: BIC MODEL SELECTION
# ============================================================


def plot_bic_panel(ax, bic_table, selected_k, selection_label):
    """Plot BIC scores and highlight the selected component number."""
    ax.plot(
        bic_table["K"],
        bic_table["BIC"],
        color="#2878B5",
        linewidth=1.15,
        marker="o",
        markersize=3.0,
        markerfacecolor="#2878B5",
        markeredgewidth=0.45,
    )

    selected_bic = float(bic_table.loc[bic_table["K"] == selected_k, "BIC"].iloc[0])

    ax.scatter(
        [selected_k],
        [selected_bic],
        s=25,
        color="#D7191C",
        zorder=5,
    )

    ax.axvline(
        selected_k,
        color="#D7191C",
        linestyle="--",
        linewidth=0.8,
        label=selection_label,
    )

    ax.set_xlabel("Number of Gaussian Components (K)")
    ax.set_ylabel("BIC Score")
    ax.set_xticks(bic_table["K"].tolist())
    ax.set_xlim(
        bic_table["K"].min() - 0.35,
        bic_table["K"].max() + 0.35,
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.2,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.45,
        alpha=0.35,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 8. PLOT PANEL B: GMM DISTRIBUTIONS AND THRESHOLDS
# ============================================================


def plot_gmm_panel(
    ax,
    values,
    weights,
    means,
    std_devs,
    thresholds,
    grade_names,
    grade_colors,
):
    """Plot empirical histogram, weighted Gaussian curves and thresholds."""
    x_upper = max(
        float(values.max()) * 1.05,
        float(np.max(means + 3.0 * std_devs)),
    )
    x_upper = min(1.0, max(x_upper, 0.1))

    x_grid = np.linspace(0.0, x_upper, 2500)

    ax.hist(
        values,
        bins=55,
        density=True,
        color="#D2D2D2",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
        label="Empirical distribution",
        zorder=1,
    )

    for i in range(len(weights)):
        density = weights[i] * norm.pdf(
            x_grid,
            loc=means[i],
            scale=std_devs[i],
        )

        ax.plot(
            x_grid,
            density,
            color=grade_colors[i],
            linewidth=1.25,
            label=grade_names[i],
            zorder=3,
        )

    # Threshold lines and labels.
    label_y_positions = np.linspace(0.92, 0.68, max(len(thresholds), 1))

    for i, threshold in enumerate(thresholds):
        ax.axvline(
            threshold,
            color="#555555",
            linestyle=":",
            linewidth=0.75,
            zorder=2,
        )

        ax.text(
            threshold + x_upper * 0.008,
            label_y_positions[i],
            f"T{i + 1}={threshold:.3f}",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_SIZE_PT,
            ha="left",
            va="center",
            color="#333333",
        )

    ax.set_xlim(0, x_upper)
    ax.set_xlabel("Comprehensive Yellow Score (CYS)")
    ax.set_ylabel("Probability Density")

    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=1,
        handlelength=1.5,
        labelspacing=0.25,
        borderaxespad=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 9. PLOT PANEL C: VIOLIN PLOT WITH OVERLAID BOXPLOTS
# ============================================================


def plot_violin_panel(ax, graded_df, summary, grade_names, grade_colors):
    """Plot CYS distributions by grade using violins and internal boxplots."""
    grade_codes = summary["Grade"].astype(int).tolist()

    groups = []
    positions = []

    for grade in grade_codes:
        group = (
            graded_df.loc[
                graded_df["CYS_GMM_Grade"] == grade,
                TARGET_COL,
            ]
            .astype(float)
            .to_numpy()
        )

        groups.append(group)
        positions.append(grade)

    # Violin distributions.
    nonempty_groups = [
        (pos, group) for pos, group in zip(positions, groups) if len(group) >= 2
    ]

    if nonempty_groups:
        violin_positions = [item[0] for item in nonempty_groups]
        violin_data = [item[1] for item in nonempty_groups]

        violin = ax.violinplot(
            violin_data,
            positions=violin_positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=160,
            bw_method="scott",
        )

        for body, pos in zip(violin["bodies"], violin_positions):
            body.set_facecolor(grade_colors[pos - 1])
            body.set_edgecolor("#666666")
            body.set_linewidth(0.55)
            body.set_alpha(0.90)

    # Overlay compact white boxplots.
    nonempty_box_data = [
        (pos, group) for pos, group in zip(positions, groups) if len(group) > 0
    ]

    if nonempty_box_data:
        box_positions = [item[0] for item in nonempty_box_data]
        box_data = [item[1] for item in nonempty_box_data]

        box = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.17,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            medianprops={
                "color": "#333333",
                "linewidth": 0.85,
            },
            boxprops={
                "edgecolor": "#555555",
                "linewidth": 0.65,
            },
            whiskerprops={
                "color": "#555555",
                "linewidth": 0.65,
            },
            capprops={
                "color": "#555555",
                "linewidth": 0.65,
            },
        )

        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.95)

    # Sample-size labels above each grade distribution.
    max_value = float(graded_df[TARGET_COL].max())
    y_upper = min(1.0, max(0.8, max_value * 1.16 + 0.04))
    y_offset = y_upper * 0.035

    for pos, group in zip(positions, groups):
        if len(group) == 0:
            continue

        label_y = min(float(np.max(group)) + y_offset, y_upper * 0.97)

        ax.text(
            pos,
            label_y,
            f"n={len(group)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_PT,
            color="#555555",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        grade_names,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_xlim(0.45, len(positions) + 0.55)
    ax.set_ylim(0, y_upper)
    ax.set_ylabel("CYS")

    # Total sample size, matching the reference figure style.
    ax.text(
        0.98,
        0.045,
        f"n = {len(graded_df):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_PT,
        fontweight="bold",
        color="#222222",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 10. CREATE THE THREE-PANEL FIGURE
# ============================================================


def create_figure(
    values,
    bic_table,
    selected_k,
    selection_label,
    weights,
    means,
    std_devs,
    thresholds,
    graded_df,
    summary,
    grade_names,
    output_dir,
):
    """Create and save the A-C figure as editable SVG and high-resolution PNG."""
    if selected_k == 5:
        grade_colors = GRADE_COLORS.copy()
    else:
        # Use a continuous color sequence for non-five-grade solutions.
        cmap = plt.get_cmap("YlOrBr")
        grade_colors = [
            cmap(0.18 + 0.72 * i / max(selected_k - 1, 1)) for i in range(selected_k)
        ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(
            FIG_WIDTH_CM * CM_TO_INCH,
            FIG_HEIGHT_CM * CM_TO_INCH,
        ),
        gridspec_kw={
            "height_ratios": [0.82, 1.18, 0.98],
        },
    )

    # Panel A
    plot_bic_panel(
        axes[0],
        bic_table,
        selected_k,
        selection_label,
    )

    # Panel B
    plot_gmm_panel(
        axes[1],
        values,
        weights,
        means,
        std_devs,
        thresholds,
        grade_names,
        grade_colors,
    )

    # Panel C
    plot_violin_panel(
        axes[2],
        graded_df,
        summary,
        grade_names,
        grade_colors,
    )

    # Panel labels.
    for ax, label in zip(axes, ["(A)", "(B)", "(C)"]):
        ax.text(
            -0.16,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=FONT_SIZE_PT,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )

    # Layout for a narrow left-hand column in a 17-cm composite figure.
    fig.subplots_adjust(
        left=0.19,
        right=0.98,
        top=0.985,
        bottom=0.075,
        hspace=0.56,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / "CYS_GMM_ABC_ThreePanels.svg"
    png_path = output_dir / "CYS_GMM_ABC_ThreePanels.png"

    # SVG text remains editable; plotted curves, axes and markers are vector.
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )

    if SAVE_PNG:
        fig.savefig(
            png_path,
            format="png",
            dpi=PNG_DPI,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)

    print("\nFigure files saved:")
    print(f"SVG: {svg_path}")
    if SAVE_PNG:
        print(f"PNG: {png_path}")

    return svg_path, png_path


# ============================================================
# 11. EXPORT EXCEL RESULTS
# ============================================================


def export_excel(
    graded_df,
    bic_table,
    threshold_table,
    summary,
    weights,
    means,
    std_devs,
    output_dir,
):
    """Export graded observations and all key GMM statistics to Excel."""
    component_table = pd.DataFrame(
        {
            "Ordered_component": np.arange(1, len(weights) + 1),
            "Mixture_weight": weights,
            "Mean_CYS": means,
            "SD_CYS": std_devs,
        }
    )

    excel_path = output_dir / "CYS_GMM_Analysis_Results.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        graded_df.to_excel(
            writer,
            sheet_name="GMM_graded_data",
            index=False,
        )

        bic_table.to_excel(
            writer,
            sheet_name="BIC_model_selection",
            index=False,
        )

        component_table.to_excel(
            writer,
            sheet_name="GMM_components",
            index=False,
        )

        threshold_table.to_excel(
            writer,
            sheet_name="GMM_thresholds",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="Grade_summary",
            index=False,
        )

    print(f"\nExcel results saved:\n{excel_path}")
    return excel_path


# ============================================================
# 12. MAIN WORKFLOW
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and visualize the study CYS Gaussian mixture model."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--components",
        type=int,
        help="Force a component count instead of using the minimum BIC.",
    )
    args = parser.parse_args()
    print("=" * 68)
    print("CYS GMM analysis: BIC selection, thresholds, and violin plot")
    print("=" * 68)

    # Read and clean data.
    input_df = read_input_table(args.input)
    clean_df, values = clean_cys_data(input_df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit GMM models and select K.
    (
        models,
        bic_scores,
        automatic_k,
        selected_k,
        selection_label,
        selected_model,
        bic_table,
    ) = fit_gmm_models(values, args.components)

    # Sort components from low to high CYS.
    weights, means, std_devs = get_sorted_gmm_parameters(selected_model)

    print("\nOrdered GMM components:")
    for i, (weight, mean, sd) in enumerate(zip(weights, means, std_devs), start=1):
        print(f"Component {i}: " f"weight={weight:.4f}, mean={mean:.6f}, SD={sd:.6f}")

    # Calculate density intersections.
    print("\nCalculating adjacent-component thresholds...")
    thresholds, threshold_table = calculate_thresholds(
        weights,
        means,
        std_devs,
    )

    # Assign grades and calculate descriptive statistics.
    graded_df, summary, grade_names = assign_grades(
        clean_df,
        values,
        thresholds,
        selected_k,
    )

    print("\nGrade summary:")
    print(summary.to_string(index=False))

    # Create A-C figure.
    create_figure(
        values,
        bic_table,
        selected_k,
        selection_label,
        weights,
        means,
        std_devs,
        thresholds,
        graded_df,
        summary,
        grade_names,
        output_dir,
    )

    # Export Excel workbook.
    export_excel(
        graded_df,
        bic_table,
        threshold_table,
        summary,
        weights,
        means,
        std_devs,
        output_dir,
    )

    print("\n" + "=" * 68)
    print("Analysis completed.")
    print(f"Automatic BIC-selected K = {automatic_k}")
    print(f"K used for grading = {selected_k}")
    print(f"Number of valid CYS observations = {len(values)}")
    print(f"Output directory = {output_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
