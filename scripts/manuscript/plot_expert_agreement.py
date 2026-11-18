"""Reproduce the five-expert visual-agreement figure."""

from argparse import ArgumentParser
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

# ============================================================
# Latest Excel List
# ============================================================

EXPERT_COLS = [
    "Expert 1",
    "Expert 2",
    "Expert 3",
    "Expert 4",
    "Expert 5"
]

EXPERT_LABELS = [
    "Expert 1",
    "Expert 2",
    "Expert 3",
    "Expert 4",
    "Expert 5"
]

# Median of five expert ratings
REFERENCE_COL = "Final_Grade"

# Continuous yellow comprehensive score
CYS_COL = "CYS"

# CYSFive-level classification results
ALGORITHM_COL = "Scientific_Grade"

# Picture name
IMAGE_COL = "Image_name"

GRADES = [1, 2, 3, 4, 5]
GRADE_LABELS = ["G1", "G2", "G3", "G4", "G5"]


# ============================================================
# 2. FIGURE PARAMETERS
# ============================================================

FIG_WIDTH_CM = 17.0
FIG_HEIGHT_CM = 7.5

DPI = 600
FONT_SIZE = 7.5

# Five expert colors
BAR_COLORS = [
    "#8EA4C9",
    "#DFA0A2",
    "#88B9A5",
    "#EABF91",
    "#B99BC1"
]

# Pie chart colors
PIE_COLORS = [
    "#8EA4C9",
    "#88B9A5",
    "#F0D18F",
    "#EABF91",
    "#B99BC1"
]

# Confusion matrix color gradient
CMAP = LinearSegmentedColormap.from_list(
    "HPJ_Blue",
    [
        "#F3F6F9",
        "#DCE7EF",
        "#B6CFDF",
        "#85AEC6",
        "#4E86A8",
        "#234F78"
    ]
)

TEXT_COLOR = "#303943"
GRID_COLOR = "#E7ECF1"

AXIS_LINE_WIDTH = 0.75
TICK_LINE_WIDTH = 0.65


# ============================================================
# 3. GLOBAL STYLE
# ============================================================

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": FONT_SIZE,

    "axes.labelsize": 7.8,
    "axes.titlesize": 8.0,

    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,

    "legend.fontsize": 6.2,

    "axes.linewidth": AXIS_LINE_WIDTH,
    "axes.edgecolor": TEXT_COLOR,

    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,

    "xtick.major.width": TICK_LINE_WIDTH,
    "ytick.major.width": TICK_LINE_WIDTH,

    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,

    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI
})


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def add_panel_label(fig, x, label):

    fig.text(
        x, 0.965, label,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT_COLOR
    )


def validate_grade_columns(data, columns):

    for col in columns:

        values = data[col].dropna().to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values found in column: {col}"
            )

        if not np.all(np.isclose(values, np.round(values))):
            raise ValueError(
                f"Non-integer grade values found in column: {col}"
            )

        if not np.isin(
            np.round(values).astype(int), GRADES
        ).all():

            raise ValueError(
                f"Grades in column {col} must be integers from 1 to 5."
            )

        data.loc[data[col].notna(), col] = np.round(
            data.loc[data[col].notna(), col]
        ).astype(int)

    return data


def format_p_value(p_value):

    if p_value < 0.001:
        return "p < 0.001"

    return f"p = {p_value:.4g}"


def check_reference_grade(data):

    """
    Check whether Final_Grade equals the median
    of the five independent expert grades.
    """

    calculated_median = (
        data[EXPERT_COLS].median(axis=1).astype(int)
    )

    recorded_grade = data[REFERENCE_COL].astype(int)

    mismatch = calculated_median != recorded_grade

    if mismatch.any():

        print("\nERROR: Final_Grade does not match expert median!")

        print(
            data.loc[
                mismatch,
                [IMAGE_COL] + EXPERT_COLS + [REFERENCE_COL]
            ].head(10).to_string(index=False)
        )

        raise ValueError(
            "Please check Final_Grade before continuing."
        )

    print(
        "Final_Grade verification: "
        "All records match the median of five expert grades."
    )


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Workbook with five expert grades and CYS.")
    parser.add_argument("output_dir", type=Path, help="Directory for SVG, PDF, and PNG outputs.")
    args = parser.parse_args()
    input_file, output_dir = args.input, args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 5.1 Load data
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HPJ | FIVE-EXPERT CYS VALIDATION")
    print("=" * 65)

    print("\nLoading Excel data...")

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_excel(
        input_file,
        engine="openpyxl"
    )

    df.columns = df.columns.astype(str).str.strip()

    required_columns = (
        EXPERT_COLS
        + [REFERENCE_COL, CYS_COL, ALGORITHM_COL]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Convert columns to numeric
    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 5.2 Prepare data
    # --------------------------------------------------------

    # Figure data
    figure_data = df.dropna(
        subset=EXPERT_COLS + [
            REFERENCE_COL,
            ALGORITHM_COL
        ]
    ).copy()

    # Spearman data
    correlation_data = df.dropna(
        subset=EXPERT_COLS + [
            REFERENCE_COL,
            CYS_COL
        ]
    ).copy()

    if figure_data.empty:

        raise ValueError(
            "No valid records for figure generation."
        )

    if correlation_data.empty:

        raise ValueError(
            "No valid records for Spearman analysis."
        )

    figure_data = validate_grade_columns(
        figure_data,
        EXPERT_COLS + [
            REFERENCE_COL,
            ALGORITHM_COL
        ]
    )

    correlation_data = validate_grade_columns(
        correlation_data,
        EXPERT_COLS + [REFERENCE_COL]
    )

    # Check Final_Grade against the five-expert median
    check_reference_grade(figure_data)
    check_reference_grade(correlation_data)

    # Ensure CYS values are finite
    correlation_data = correlation_data[
        np.isfinite(correlation_data[CYS_COL])
    ].copy()

    if len(correlation_data) < 3:

        raise ValueError(
            "At least three valid observations are required."
        )

    n_images = len(figure_data)
    n_spearman = len(correlation_data)

    print(f"\nTotal Excel records: {len(df)}")
    print(f"Valid images for figure: {n_images}")
    print(f"Valid records for Spearman: {n_spearman}")

    if n_images != n_spearman:

        print(
            "\nWARNING: Figure and Spearman sample sizes differ. "
            "Please check missing values."
        )

    # --------------------------------------------------------
    # 5.3 Inter-expert agreement (Panel A)
    # --------------------------------------------------------

    scores = figure_data[
        EXPERT_COLS
    ].to_numpy(dtype=int)

    max_agreements = np.array([
        Counter(row).most_common(1)[0][1]
        for row in scores
    ])

    agreement_levels = [5, 4, 3, 2, 1]

    agreement_counts = np.array([
        np.sum(max_agreements == level)
        for level in agreement_levels
    ])

    agreement_labels = [
        "5/5 agreement",
        "4/5 agreement",
        "3/5 agreement",
        "2/5 agreement",
        "1/5 agreement"
    ]

    print("\nInter-expert agreement:")

    for label, count in zip(
        agreement_labels,
        agreement_counts
    ):

        if count > 0:

            print(
                f"  {label}: {count} "
                f"({count / n_images * 100:.1f}%)"
            )

    at_least_three = int(
        np.sum(max_agreements >= 3)
    )

    print(
        f"  At least 3 experts agreed: "
        f"{at_least_three} "
        f"({at_least_three / n_images * 100:.1f}%)"
    )

    # --------------------------------------------------------
    # 5.4 Confusion matrix and weighted Cohen's kappa
    # --------------------------------------------------------

    reference_grades = figure_data[
        REFERENCE_COL
    ].to_numpy(dtype=int)

    algorithm_grades = figure_data[
        ALGORITHM_COL
    ].to_numpy(dtype=int)

    cm = confusion_matrix(
        reference_grades,
        algorithm_grades,
        labels=GRADES
    )

    kappa_quadratic = cohen_kappa_score(
        reference_grades,
        algorithm_grades,
        weights="quadratic"
    )

    print("\nConfusion matrix:")
    print(cm)

    print(
        f"\nQuadratically weighted Cohen's kappa: "
        f"{kappa_quadratic:.4f}"
    )

    exact_match = int(np.trace(cm))

    total_cm = int(cm.sum())

    row_idx, col_idx = np.indices(cm.shape)

    grade_difference = np.abs(
        row_idx - col_idx
    )

    within_one = int(
        cm[grade_difference <= 1].sum()
    )

    two_or_more = int(
        cm[grade_difference >= 2].sum()
    )

    # Rows: expert reference grades
    # Columns: CYS-based grades

    lower_count = int(
        cm[np.tril_indices(5, k=-1)].sum()
    )

    higher_count = int(
        cm[np.triu_indices(5, k=1)].sum()
    )

    print(
        "\nCYS-based grade agreement with "
        "median expert grade:"
    )

    print(
        f"  Exact match: {exact_match}/{total_cm} "
        f"({exact_match / total_cm * 100:.1f}%)"
    )

    print(
        f"  Within one grade: {within_one}/{total_cm} "
        f"({within_one / total_cm * 100:.1f}%)"
    )

    print(
        f"  Difference of two or more grades: "
        f"{two_or_more}/{total_cm} "
        f"({two_or_more / total_cm * 100:.1f}%)"
    )

    print(
        f"  CYS lower than expert reference: "
        f"{lower_count}/{total_cm} "
        f"({lower_count / total_cm * 100:.1f}%)"
    )

    print(
        f"  CYS higher than expert reference: "
        f"{higher_count}/{total_cm} "
        f"({higher_count / total_cm * 100:.1f}%)"
    )

    # --------------------------------------------------------
    # 5.5 Sensitivity analysis
    # At least 3/5 experts agreed
    # --------------------------------------------------------

    agreement_subset_mask = (
        max_agreements >= 3
    )

    agreement_subset = figure_data.loc[
        agreement_subset_mask
    ].copy()

    if not agreement_subset.empty:

        # Majority-vote reference grade
        majority_reference = np.array([

            Counter(row).most_common(1)[0][0]

            for row in agreement_subset[
                EXPERT_COLS
            ].to_numpy(dtype=int)

        ], dtype=int)

        algorithm_subset = agreement_subset[
            ALGORITHM_COL
        ].to_numpy(dtype=int)

        subset_cm = confusion_matrix(
            majority_reference,
            algorithm_subset,
            labels=GRADES
        )

        subset_n = int(subset_cm.sum())

        subset_exact = int(
            np.trace(subset_cm)
        )

        subset_row_idx, subset_col_idx = np.indices(
            subset_cm.shape
        )

        subset_difference = np.abs(
            subset_row_idx - subset_col_idx
        )

        subset_within_one = int(
            subset_cm[
                subset_difference <= 1
            ].sum()
        )

        subset_two_or_more = int(
            subset_cm[
                subset_difference >= 2
            ].sum()
        )

        subset_kappa = cohen_kappa_score(
            majority_reference,
            algorithm_subset,
            weights="quadratic"
        )

        print(
            "\nSensitivity analysis "
            "(images with agreement by at least 3/5 experts):"
        )

        print(f"  Sample size: {subset_n}")

        print(
            f"  Exact match: {subset_exact}/{subset_n} "
            f"({subset_exact / subset_n * 100:.2f}%)"
        )

        print(
            f"  Within one grade: "
            f"{subset_within_one}/{subset_n} "
            f"({subset_within_one / subset_n * 100:.2f}%)"
        )

        print(
            f"  Difference of two or more grades: "
            f"{subset_two_or_more}/{subset_n} "
            f"({subset_two_or_more / subset_n * 100:.2f}%)"
        )

        print(
            f"  Quadratically weighted Cohen's kappa: "
            f"{subset_kappa:.4f}"
        )

    else:

        print(
            "\nSensitivity analysis skipped: "
            "No images with agreement by at least 3/5 experts."
        )

    # --------------------------------------------------------
    # 5.6 Expert grade distributions (Panel B)
    # --------------------------------------------------------

    expert_counts = np.array([

        figure_data[col]
        .value_counts()
        .reindex(GRADES, fill_value=0)
        .values

        for col in EXPERT_COLS

    ])

    print("\nExpert grade distributions (G1-G5):")

    for label, counts in zip(
        EXPERT_LABELS,
        expert_counts
    ):

        print(
            f"  {label}: {counts.tolist()}"
        )

    # --------------------------------------------------------
    # 5.7 Spearman correlation
    # Continuous CYS vs. median expert grade
    # --------------------------------------------------------

    correlation_reference = correlation_data[
        REFERENCE_COL
    ].to_numpy(dtype=int)

    continuous_cys = correlation_data[
        CYS_COL
    ].to_numpy(dtype=float)

    rho, p_value = spearmanr(
        continuous_cys,
        correlation_reference
    )

    if not np.isfinite(rho) or not np.isfinite(p_value):

        raise ValueError(
            "Spearman correlation could not be calculated. "
            "Check whether CYS or expert grades are constant."
        )

    print("\n" + "-" * 65)

    print(
        "SPEARMAN CORRELATION: "
        "CONTINUOUS CYS vs. MEDIAN EXPERT GRADE"
    )

    print("-" * 65)

    print(f"  Sample size (n):       {n_spearman}")
    print(f"  Spearman's rho:        {rho:.4f}")
    print(f"  P value:               {format_p_value(p_value)}")

    print("-" * 65)

    # --------------------------------------------------------
    # 5.8 Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            FIG_WIDTH_CM / 2.54,
            FIG_HEIGHT_CM / 2.54
        )
    )

    # Panel positions
    ax_a = fig.add_axes(
        [0.035, 0.425, 0.195, 0.455]
    )

    ax_b = fig.add_axes(
        [0.375, 0.225, 0.355, 0.600]
    )

    ax_c = fig.add_axes(
        [0.800, 0.285, 0.185, 0.515]
    )

    # ========================================================
    # PANEL A: INTER-EXPERT AGREEMENT
    # ========================================================

    plot_mask = agreement_counts > 0

    pie_values = agreement_counts[plot_mask]

    pie_colors = np.array(
        PIE_COLORS
    )[plot_mask]

    pie_labels = np.array(
        agreement_labels
    )[plot_mask]

    def count_label(pct):

        count = int(
            round(pct * pie_values.sum() / 100)
        )

        return str(count) if count > 0 else ""

    wedges, _, autotexts = ax_a.pie(
        pie_values,
        colors=pie_colors,
        startangle=90,
        counterclock=False,
        radius=0.96,
        autopct=count_label,
        pctdistance=0.63,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1.25
        },
        textprops={
            "fontsize": 7.2,
            "color": TEXT_COLOR
        }
    )

    for txt in autotexts:
        txt.set_fontweight("medium")

    ax_a.set_aspect("equal")

    pie_legend_labels = [

        f"{label} (n = {value}, "
        f"{value / n_images * 100:.1f}%)"

        for label, value in zip(
            pie_labels,
            pie_values
        )

    ]

    ax_a.legend(
        wedges,
        pie_legend_labels,
        title="Inter-expert agreement (five experts)",
        title_fontsize=5.9,
        fontsize=5.7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        frameon=False,
        ncol=1,
        handlelength=0.9,
        handleheight=0.65,
        handletextpad=0.4,
        labelspacing=0.38,
        borderaxespad=0
    )

    # ========================================================
    # PANEL B: FIVE-EXPERT GRADE DISTRIBUTION
    # ========================================================

    x = np.arange(len(GRADES))

    bar_width = 0.14
    bar_step = 0.145

    offsets = (
        np.arange(len(EXPERT_COLS)) - 2
    ) * bar_step

    label_offsets = [
        2.0, 5.0, 2.0, 5.0, 2.0
    ]

    for i, (label, color) in enumerate(
        zip(EXPERT_LABELS, BAR_COLORS)
    ):

        counts = expert_counts[i]

        xpos = x + offsets[i]

        ax_b.bar(
            xpos,
            counts,
            width=bar_width,
            bottom=0,
            color=color,
            edgecolor="none",
            label=label,
            align="center",
            zorder=2
        )

        for j, height in enumerate(counts):

            if height <= 0:
                continue

            ax_b.text(
                xpos[j],
                height + label_offsets[i],
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=5.4,
                color=TEXT_COLOR,
                fontweight="normal",
                clip_on=False,
                zorder=4
            )

    ax_b.set_xticks(x)

    ax_b.set_xticklabels(GRADE_LABELS)

    ax_b.set_xlim(-0.50, 4.50)

    max_count = int(expert_counts.max())

    y_upper = max(
        115,
        int(np.ceil((max_count + 15) / 10) * 10)
    )

    ax_b.set_ylim(
        bottom=0,
        top=y_upper
    )

    ax_b.set_xlabel(
        "Expert-assigned grade",
        labelpad=4
    )

    ax_b.set_ylabel(
        "Number of images",
        labelpad=4
    )

    ax_b.yaxis.set_major_locator(
        MultipleLocator(25)
    )

    ax_b.set_axisbelow(True)

    ax_b.yaxis.grid(
        True,
        color=GRID_COLOR,
        linewidth=0.55,
        linestyle="-",
        zorder=0
    )

    ax_b.xaxis.grid(False)

    ax_b.spines["bottom"].set_visible(True)

    ax_b.spines["bottom"].set_position(
        ("axes", 0)
    )

    ax_b.spines["bottom"].set_color(
        TEXT_COLOR
    )

    ax_b.spines["bottom"].set_linewidth(
        AXIS_LINE_WIDTH
    )

    ax_b.spines["bottom"].set_zorder(10)

    ax_b.spines["left"].set_zorder(10)

    ax_b.spines["left"].set_linewidth(
        AXIS_LINE_WIDTH
    )

    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    ax_b.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=2.8,
        width=TICK_LINE_WIDTH,
        pad=2,
        bottom=True,
        top=False
    )

    ax_b.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=2.8,
        width=TICK_LINE_WIDTH,
        pad=3
    )

    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=5,
        frameon=False,
        fontsize=6.0,
        handlelength=0.9,
        handleheight=0.65,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0
    )

    # ========================================================
    # PANEL C: CONFUSION MATRIX
    # ========================================================

    vmax = max(
        1,
        int(cm.max())
    )

    ax_c.imshow(
        cm,
        cmap=CMAP,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    threshold = vmax * 0.50

    for i in range(5):

        for j in range(5):

            value = int(cm[i, j])

            ax_c.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="semibold",
                color=(
                    "white"
                    if value > threshold
                    else TEXT_COLOR
                )
            )

    ax_c.set_xticks(range(5))
    ax_c.set_yticks(range(5))

    ax_c.set_xticklabels(
        GRADE_LABELS,
        fontsize=6.5
    )

    ax_c.set_yticklabels(
        GRADE_LABELS,
        fontsize=6.5
    )

    ax_c.set_xlabel(
        "CYS-based grade",
        labelpad=4
    )

    ax_c.set_ylabel(
        "Median expert grade",
        labelpad=4
    )

    ax_c.set_xticks(
        np.arange(-0.5, 5, 1),
        minor=True
    )

    ax_c.set_yticks(
        np.arange(-0.5, 5, 1),
        minor=True
    )

    ax_c.grid(
        which="minor",
        color="white",
        linewidth=1.2
    )

    ax_c.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax_c.tick_params(
        which="major",
        length=0,
        pad=3
    )

    for spine in ax_c.spines.values():
        spine.set_visible(False)

    # ========================================================
    # 6. PANEL LABELS
    # ========================================================

    add_panel_label(fig, 0.015, "(A)")
    add_panel_label(fig, 0.365, "(B)")
    add_panel_label(fig, 0.790, "(C)")

    # ========================================================
    # 7. SAVE FIGURES
    # ========================================================

    prefix = output_dir / "Fig_Expert_Validation_Combined"

    fig.savefig(
        str(prefix) + ".svg",
        format="svg"
    )

    fig.savefig(
        str(prefix) + ".pdf",
        format="pdf"
    )

    fig.savefig(
        str(prefix) + f"_{DPI}dpi.png",
        format="png",
        dpi=DPI
    )

    # ========================================================
    # 8. FINAL TERMINAL REPORT
    # ========================================================

    print("\n" + "=" * 65)
    print("FIGURE SUCCESSFULLY GENERATED")
    print("=" * 65)

    print(f"Images used for figure: {n_images}")
    print(f"Images used for Spearman: {n_spearman}")

    print(f"Spearman's rho: {rho:.4f}")
    print(f"P value: {format_p_value(p_value)}")

    print(
        f"Quadratically weighted Cohen's kappa: "
        f"{kappa_quadratic:.4f}"
    )

    print(f"SVG: {prefix}.svg")
    print(f"PDF: {prefix}.pdf")
    print(f"PNG: {prefix}_{DPI}dpi.png")

    print("=" * 65)

    plt.close(fig)


if __name__ == "__main__":
    main()
