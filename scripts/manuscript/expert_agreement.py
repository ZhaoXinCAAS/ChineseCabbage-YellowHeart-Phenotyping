"""Reproduce the five-expert agreement and CYS validation workbook."""

from argparse import ArgumentParser
from pathlib import Path, PureWindowsPath
from itertools import combinations
from collections import Counter
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    rankdata,
    chi2,
    spearmanr,
    binomtest,
    wilcoxon,
)

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

IMAGE_COL = "Image_name"

EXPERT_COLS = [
    "Expert 1",
    "Expert 2",
    "Expert 3",
    "Expert 4",
    "Expert 5",
]

FINAL_GRADE_COL = "Final_Grade"
CYS_COL = "CYS"
ALGORITHM_COL = "Scientific_Grade"

GRADES = [1, 2, 3, 4, 5]
GRADE_NAMES = {
    1: "G1",
    2: "G2",
    3: "G3",
    4: "G4",
    5: "G5",
}



# ============================================================
# 2. GENERAL FUNCTIONS
# ============================================================

def p_text(p):
    """Readable P-value format."""
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.4f}"


def as_numeric(df, columns):
    """Convert specified columns to numeric; invalid cells become NaN."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_ordinal_grades(df, columns):
    """Validate non-missing ordinal grades as integers from 1 to 5."""
    for col in columns:
        values = df[col].dropna().to_numpy(dtype=float)

        if len(values) == 0:
            continue

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{col} contains non-finite values.")

        if not np.all(np.isclose(values, np.round(values))):
            bad = df.loc[
                df[col].notna() & ~np.isclose(df[col], np.round(df[col])),
                [IMAGE_COL, col]
            ].head(10)
            raise ValueError(
                f"{col} contains non-integer grades. Examples:\n{bad}"
            )

        if not np.isin(np.round(values).astype(int), GRADES).all():
            bad = df.loc[
                df[col].notna() & ~df[col].isin(GRADES),
                [IMAGE_COL, col]
            ].head(10)
            raise ValueError(
                f"{col} must contain grades 1-5 only. Examples:\n{bad}"
            )

        df[col] = df[col].round().astype("Int64")

    return df


def quadratic_weighted_kappa(a, b):
    """Quadratically weighted Cohen's kappa for 5 ordered categories."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)

    if len(a) == 0:
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            cohen_kappa_score(
                a,
                b,
                labels=GRADES,
                weights="quadratic"
            )
        )


def calculate_kendall_w(rating_matrix):
    """
    Kendall's coefficient of concordance W with tie correction.

    Rows = images; columns = experts.
    Each expert ranks the images. Tied grades receive average ranks.
    """
    rating_matrix = np.asarray(rating_matrix, dtype=float)
    n, m = rating_matrix.shape

    if n < 2 or m < 2:
        raise ValueError("Kendall's W requires at least 2 images and 2 experts.")

    # Rank images separately for each expert.
    ranks = np.column_stack([
        rankdata(rating_matrix[:, j], method="average")
        for j in range(m)
    ])

    rank_sums = ranks.sum(axis=1)
    expected_rank_sum = m * (n + 1) / 2.0
    S = np.sum((rank_sums - expected_rank_sum) ** 2)

    # Tie correction: sum_j sum_g (t_jg^3 - t_jg)
    tie_sum = 0.0
    for j in range(m):
        _, counts = np.unique(rating_matrix[:, j], return_counts=True)
        tie_sum += np.sum(counts ** 3 - counts)

    denominator = (m ** 2) * (n ** 3 - n) - m * tie_sum

    if denominator <= 0:
        raise ValueError(
            "Kendall's W cannot be calculated: insufficient rating variation."
        )

    W = float(np.clip(12.0 * S / denominator, 0.0, 1.0))

    # Large-sample chi-square approximation commonly used for Kendall's W.
    chi_square = m * (n - 1) * W
    df_chi = n - 1
    p_value = float(chi2.sf(chi_square, df_chi))

    return W, float(chi_square), int(df_chi), p_value


def agreement_metrics(reference, predicted):
    """Agreement and direction-of-discrepancy metrics for ordinal grades."""
    reference = np.asarray(reference, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    if len(reference) != len(predicted):
        raise ValueError("Reference and predicted arrays have different lengths.")
    if len(reference) == 0:
        raise ValueError("No paired observations are available.")

    diff = predicted - reference
    abs_diff = np.abs(diff)
    n = len(diff)

    exact = int(np.sum(abs_diff == 0))
    within_one = int(np.sum(abs_diff <= 1))
    two_or_more = int(np.sum(abs_diff >= 2))
    lower = int(np.sum(diff < 0))
    equal = int(np.sum(diff == 0))
    higher = int(np.sum(diff > 0))

    return {
        "Sample size": n,
        "Exact agreement n": exact,
        "Exact agreement (%)": exact / n * 100,
        "Within-one-grade agreement n": within_one,
        "Within-one-grade agreement (%)": within_one / n * 100,
        "Difference >=2 grades n": two_or_more,
        "Difference >=2 grades (%)": two_or_more / n * 100,
        "Algorithm/reference lower n": lower,
        "Algorithm/reference lower (%)": lower / n * 100,
        "Equal n": equal,
        "Equal (%)": equal / n * 100,
        "Algorithm/reference higher n": higher,
        "Algorithm/reference higher (%)": higher / n * 100,
        "Mean signed difference (predicted-reference)": float(np.mean(diff)),
        "Median signed difference (predicted-reference)": float(np.median(diff)),
        "Mean absolute difference": float(np.mean(abs_diff)),
        "Median absolute difference": float(np.median(abs_diff)),
        "Quadratically weighted Cohen kappa": quadratic_weighted_kappa(
            reference, predicted
        ),
    }


def image_keys(names, label):
    """Match single-head filenames across image formats, retaining the head suffix."""
    keys = []
    for name in names.astype("string"):
        if pd.isna(name) or not name.strip():
            raise ValueError(f"{label} contains a missing image filename.")
        path = PureWindowsPath(name.strip())
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        key = path.stem if path.suffix.casefold() in extensions else path.name
        keys.append(key.casefold())
    keys = pd.Series(keys, index=names.index, dtype="string")
    duplicates = keys[keys.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate normalized image filenames in {label}: {duplicates[:8]}")
    return keys


def fisher_overlap_results(algorithm_data, fisher_path):
    fisher = pd.read_excel(fisher_path, engine="openpyxl", usecols=["filename"])
    fisher_keys = image_keys(fisher["filename"], "Fisher feature workbook")
    if fisher_keys.empty:
        raise ValueError("Fisher feature workbook has no image filenames.")
    images = algorithm_data[
        [IMAGE_COL, CYS_COL, ALGORITHM_COL, "Median_Grade_Calculated"]
    ].copy()
    images["Filename_key"] = image_keys(images[IMAGE_COL], "expert comparison data")
    images["Used_for_Fisher_weighting"] = images["Filename_key"].isin(fisher_keys)

    rows = []
    for used, cohort in images.groupby("Used_for_Fisher_weighting", sort=False):
        reference = cohort["Median_Grade_Calculated"]
        stats = agreement_metrics(reference, cohort[ALGORITHM_COL])
        rho = np.nan
        if reference.nunique() > 1 and cohort[CYS_COL].nunique() > 1:
            rho = spearmanr(cohort[CYS_COL], reference).statistic
        rows.append({
            "Used_for_Fisher_weighting": used,
            **stats,
            "Spearman rho: CYS vs expert median grade": float(rho),
        })
    return pd.DataFrame(rows), images


def max_agreement_count(row):
    """Maximum number of experts assigning the same grade to one image."""
    counts = Counter(int(x) for x in row)
    return max(counts.values())


def majority_grade_if_exists(row):
    """
    Return unique majority grade if at least 3/5 experts agree.
    Return NaN if no majority (maximum count <= 2).
    """
    counts = Counter(int(x) for x in row)
    grade, count = counts.most_common(1)[0]
    return grade if count >= 3 else np.nan


def style_workbook(writer):
    """Apply clean, readable formatting to all worksheets."""
    header_fill = PatternFill("solid", fgColor="244A64")
    header_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    thin_grey = Side(style="thin", color="D9E1E8")

    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 28

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )
            cell.border = Border(bottom=thin_grey)

        for col_cells in ws.columns:
            col_letter = get_column_letter(col_cells[0].column)
            values = [
                len(str(c.value))
                for c in col_cells
                if c.value is not None
            ]
            max_len = max(values, default=10)
            ws.column_dimensions[col_letter].width = min(
                max(max_len + 3, 13), 42
            )

            for cell in col_cells[1:]:
                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=False
                )
                if isinstance(cell.value, (float, np.floating)):
                    cell.number_format = "0.0000"


def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Workbook with five expert grades and CYS.")
    parser.add_argument("output", type=Path, help="Path for the results workbook.")
    parser.add_argument("--print-image-details", action="store_true")
    parser.add_argument(
        "--fisher-features", type=Path,
        help="Fisher weighting XLSX with a filename column for the image-overlap analysis.",
    )
    args = parser.parse_args()
    input_file, output_file = args.input, args.output
    if any(path and path.resolve() == output_file.resolve()
           for path in (input_file, args.fisher_features)):
        parser.error("Input and output workbooks must be different.")
    print_section("FIVE-EXPERT GRADING RELIABILITY AND CYS VALIDATION")

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    df = pd.read_excel(input_file, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()

    required_columns = (
        [IMAGE_COL]
        + EXPERT_COLS
        + [FINAL_GRADE_COL, CYS_COL, ALGORITHM_COL]
    )

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Columns found in the Excel file: {list(df.columns)}"
        )

    total_rows = len(df)

    # Remove only completely empty image-name rows.
    df[IMAGE_COL] = df[IMAGE_COL].astype("string").str.strip()
    empty_name = df[IMAGE_COL].isna() | (df[IMAGE_COL] == "")
    empty_name_count = int(empty_name.sum())
    df = df.loc[~empty_name].copy()

    # Duplicate image names should be resolved before analysis.
    duplicate_mask = df[IMAGE_COL].duplicated(keep=False)
    duplicate_names_df = df.loc[duplicate_mask, [IMAGE_COL] + EXPERT_COLS].copy()

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate Image_name values were detected. "
            "The script stops to prevent counting the same image more than once.\n"
            f"Duplicate rows:\n{duplicate_names_df.to_string(index=False)}"
        )

    numeric_cols = EXPERT_COLS + [FINAL_GRADE_COL, CYS_COL, ALGORITHM_COL]
    df = as_numeric(df, numeric_cols)

    # Validate expert grades and algorithm grades when present.
    df = validate_ordinal_grades(df, EXPERT_COLS + [FINAL_GRADE_COL, ALGORITHM_COL])

    # --------------------------------------------------------
    # Expert reliability dataset:
    # requires all five expert grades only.
    # Do NOT discard an image just because CYS/algorithm is missing.
    # --------------------------------------------------------
    expert_complete_mask = df[EXPERT_COLS].notna().all(axis=1)
    expert_data = df.loc[expert_complete_mask].copy()

    if len(expert_data) < 3:
        raise ValueError("Fewer than 3 images have complete scores from all five experts.")

    expert_matrix = expert_data[EXPERT_COLS].astype(int).to_numpy()
    n_expert = len(expert_data)

    # Calculated five-expert median; with five integer ratings this is an integer.
    expert_data["Median_Grade_Calculated"] = (
        expert_data[EXPERT_COLS].astype(int).median(axis=1).astype(int)
    )

    # Maximum agreement count and majority grade.
    expert_data["Max_Expert_Agreement_Count"] = [
        max_agreement_count(row) for row in expert_matrix
    ]
    expert_data["Majority_Grade_3plus"] = [
        majority_grade_if_exists(row) for row in expert_matrix
    ]

    # Check the supplied Final_Grade against the median recalculated from Expert 1-5.
    final_available = expert_data[FINAL_GRADE_COL].notna()
    final_mismatch_mask = pd.Series(False, index=expert_data.index)
    final_mismatch_mask.loc[final_available] = (
        expert_data.loc[final_available, FINAL_GRADE_COL].astype(int).to_numpy()
        != expert_data.loc[final_available, "Median_Grade_Calculated"].to_numpy()
    )
    mismatch_count = int(final_mismatch_mask.sum())

    mismatch_details = expert_data.loc[
        final_mismatch_mask,
        [IMAGE_COL] + EXPERT_COLS + [
            FINAL_GRADE_COL, "Median_Grade_Calculated"
        ]
    ].copy()

    # --------------------------------------------------------
    # Algorithm validation dataset:
    # requires complete expert grades + CYS + Scientific_Grade.
    # --------------------------------------------------------
    algorithm_complete_mask = (
        expert_data[CYS_COL].notna()
        & expert_data[ALGORITHM_COL].notna()
        & np.isfinite(expert_data[CYS_COL].astype(float))
    )
    algorithm_data = expert_data.loc[algorithm_complete_mask].copy()

    if len(algorithm_data) > 0:
        algorithm_data[ALGORITHM_COL] = algorithm_data[ALGORITHM_COL].astype(int)
        algorithm_data[CYS_COL] = algorithm_data[CYS_COL].astype(float)

    n_algorithm = len(algorithm_data)
    if args.fisher_features is not None:
        if n_algorithm == 0:
            parser.error("--fisher-features requires complete expert and algorithm comparisons.")
        overlap_stats, overlap_images = fisher_overlap_results(
            algorithm_data, args.fisher_features
        )

    print(f"Input Excel rows (including blank-name rows): {total_rows}")
    print(f"Rows with empty Image_name excluded: {empty_name_count}")
    print(f"Images with complete ratings from all five experts: {n_expert}")
    print(f"Images available for CYS/algorithm comparison: {n_algorithm}")
    print(f"Images excluded from expert analysis due to missing expert score: "
          f"{len(df) - n_expert}")
    print(f"Final_Grade missing among expert-complete images: "
          f"{int(expert_data[FINAL_GRADE_COL].isna().sum())}")
    print(f"Final_Grade vs recalculated expert median mismatches: {mismatch_count}")

    if mismatch_count:
        print(
            "WARNING: Final_Grade does not match the median of Expert 1-5 "
            "for some images. The script uses the recalculated median as "
            "the primary expert reference. See Median_Mismatch_Details."
        )
    else:
        print("Final_Grade verification: PASSED for all non-missing values.")

    # ========================================================
    # 4. KENDALL'S W: OVERALL EXPERT CONCORDANCE
    # ========================================================

    W, chi_square, df_chi, p_kendall = calculate_kendall_w(expert_matrix)

    kendall_df = pd.DataFrame([{
        "Number of images": n_expert,
        "Number of experts": len(EXPERT_COLS),
        "Kendall's W": W,
        "Chi-square": chi_square,
        "Degrees of freedom": df_chi,
        "P-value": p_kendall,
        "Interpretation": (
            "Overall concordance in image ranking across the five experts; "
            "not the same as exact grade agreement."
        )
    }])

    print_section("1. OVERALL INTER-RATER CONCORDANCE: KENDALL'S W")
    print(f"Images: {n_expert}")
    print(f"Experts: {len(EXPERT_COLS)}")
    print(f"Kendall's W = {W:.4f}")
    print(f"Chi-square = {chi_square:.4f}; df = {df_chi}")
    print(f"P-value {p_text(p_kendall)}")

    # ========================================================
    # 5. PAIRWISE QUADRATICALLY WEIGHTED COHEN'S KAPPA
    # ========================================================

    pairwise_rows = []

    for expert_a, expert_b in combinations(EXPERT_COLS, 2):
        a = expert_data[expert_a].astype(int).to_numpy()
        b = expert_data[expert_b].astype(int).to_numpy()
        diff = np.abs(a - b)

        pairwise_rows.append({
            "Expert pair": f"{expert_a} vs {expert_b}",
            "n": n_expert,
            "Exact agreement n": int(np.sum(diff == 0)),
            "Exact agreement (%)": float(np.mean(diff == 0) * 100),
            "Within-one-grade agreement n": int(np.sum(diff <= 1)),
            "Within-one-grade agreement (%)": float(np.mean(diff <= 1) * 100),
            "Quadratically weighted Cohen kappa": quadratic_weighted_kappa(a, b),
        })

    pairwise_df = pd.DataFrame(pairwise_rows)
    kappa_vals = pairwise_df["Quadratically weighted Cohen kappa"].dropna()

    pairwise_summary_df = pd.DataFrame([
        {"Statistic": "Mean pairwise weighted kappa", "Value": kappa_vals.mean()},
        {"Statistic": "SD of pairwise weighted kappa", "Value": kappa_vals.std(ddof=1)},
        {"Statistic": "Minimum pairwise weighted kappa", "Value": kappa_vals.min()},
        {"Statistic": "Maximum pairwise weighted kappa", "Value": kappa_vals.max()},
        {"Statistic": "Mean pairwise exact agreement (%)",
         "Value": pairwise_df["Exact agreement (%)"].mean()},
        {"Statistic": "Mean pairwise within-one-grade agreement (%)",
         "Value": pairwise_df["Within-one-grade agreement (%)"].mean()},
    ])

    print_section("2. PAIRWISE QUADRATICALLY WEIGHTED COHEN'S KAPPA")
    for _, row in pairwise_df.iterrows():
        print(
            f"{row['Expert pair']:<28} "
            f"kappa = {row['Quadratically weighted Cohen kappa']:.4f}; "
            f"exact = {row['Exact agreement (%)']:.2f}%; "
            f"within one grade = {row['Within-one-grade agreement (%)']:.2f}%"
        )
    print(f"\nMean ± SD weighted kappa = "
          f"{kappa_vals.mean():.4f} ± {kappa_vals.std(ddof=1):.4f}")
    print(f"Mean pairwise exact agreement = "
          f"{pairwise_df['Exact agreement (%)'].mean():.2f}%")

    # ========================================================
    # 6. GRADE DISTRIBUTION FOR EACH EXPERT
    # ========================================================

    grade_count_rows = []
    for expert in EXPERT_COLS:
        counts = expert_data[expert].astype(int).value_counts().reindex(
            GRADES, fill_value=0
        )
        row = {"Expert": expert, "n": n_expert}
        for g in GRADES:
            row[f"G{g}_n"] = int(counts.loc[g])
            row[f"G{g}_percent"] = float(counts.loc[g] / n_expert * 100)
        grade_count_rows.append(row)

    expert_grade_counts_df = pd.DataFrame(grade_count_rows)

    print_section("3. GRADE DISTRIBUTION BY EXPERT")
    print(expert_grade_counts_df.to_string(index=False))

    # ========================================================
    # 7. IMAGE-LEVEL MAXIMUM EXPERT AGREEMENT
    # ========================================================

    agreement_distribution = (
        expert_data["Max_Expert_Agreement_Count"]
        .value_counts()
        .reindex([5, 4, 3, 2, 1], fill_value=0)
    )

    agreement_dist_df = pd.DataFrame({
        "Maximum number of experts assigning the same grade": [5, 4, 3, 2, 1],
        "Images (n)": [int(agreement_distribution.loc[x]) for x in [5, 4, 3, 2, 1]],
        "Percentage (%)": [
            float(agreement_distribution.loc[x] / n_expert * 100)
            for x in [5, 4, 3, 2, 1]
        ],
    })

    at_least_3 = int((expert_data["Max_Expert_Agreement_Count"] >= 3).sum())

    print_section("4. IMAGE-LEVEL INTER-EXPERT AGREEMENT")
    for _, row in agreement_dist_df.iterrows():
        print(
            f"{int(row.iloc[0])}/5 agreement: "
            f"{int(row['Images (n)'])} images "
            f"({row['Percentage (%)']:.2f}%)"
        )
    print(f"At least 3/5 experts agreed: {at_least_3}/{n_expert} "
          f"({at_least_3 / n_expert * 100:.2f}%)")

    # ========================================================
    # 8. INDIVIDUAL EXPERT VS FIVE-EXPERT MEDIAN
    # ========================================================

    # This is descriptive, not an independent validation:
    # each expert contributes to the median reference.
    median_reference = expert_data["Median_Grade_Calculated"].astype(int).to_numpy()

    expert_vs_median_rows = []
    for expert in EXPERT_COLS:
        ratings = expert_data[expert].astype(int).to_numpy()
        result = agreement_metrics(median_reference, ratings)
        expert_vs_median_rows.append({
            "Expert": expert,
            **result,
            "Interpretation": (
                "Descriptive only: this expert is included in the "
                "five-expert median reference."
            )
        })

    expert_vs_median_df = pd.DataFrame(expert_vs_median_rows)

    print_section("5. INDIVIDUAL EXPERT VS FIVE-EXPERT MEDIAN (DESCRIPTIVE)")
    print("Note: each expert contributes to the median reference; "
          "these are not independent validation comparisons.")
    for _, row in expert_vs_median_df.iterrows():
        print(
            f"{row['Expert']}: exact = {row['Exact agreement (%)']:.2f}%; "
            f"within one grade = {row['Within-one-grade agreement (%)']:.2f}%; "
            f"weighted kappa = {row['Quadratically weighted Cohen kappa']:.4f}"
        )

    # ========================================================
    # 9. CYS ALGORITHM VS FIVE-EXPERT MEDIAN
    # ========================================================

    algorithm_metrics_df = pd.DataFrame()
    confusion_df = pd.DataFrame()
    algorithm_image_df = pd.DataFrame()
    bias_test_df = pd.DataFrame()
    sensitivity_metrics_df = pd.DataFrame()
    sensitivity_confusion_df = pd.DataFrame()
    rho = np.nan
    p_spearman = np.nan

    if n_algorithm > 0:
        algo_reference = algorithm_data["Median_Grade_Calculated"].astype(int).to_numpy()
        algo_grade = algorithm_data[ALGORITHM_COL].astype(int).to_numpy()
        cys_values = algorithm_data[CYS_COL].astype(float).to_numpy()

        algo_result = agreement_metrics(algo_reference, algo_grade)

        rho, p_spearman = spearmanr(cys_values, algo_reference)

        metrics_rows = [
            {"Metric": key, "Value": value}
            for key, value in algo_result.items()
        ]
        metrics_rows.extend([
            {"Metric": "Spearman rho: CYS vs expert median grade", "Value": float(rho)},
            {"Metric": "Spearman P-value", "Value": float(p_spearman)},
        ])
        algorithm_metrics_df = pd.DataFrame(metrics_rows)

        cm = confusion_matrix(algo_reference, algo_grade, labels=GRADES)
        confusion_df = pd.DataFrame(
            cm,
            index=[f"Expert median {GRADE_NAMES[g]}" for g in GRADES],
            columns=[f"CYS {GRADE_NAMES[g]}" for g in GRADES]
        )
        confusion_df.index.name = "Expert reference grade"

        # Image-level algorithm discrepancy table.
        algorithm_image_df = algorithm_data[
            [IMAGE_COL] + EXPERT_COLS + [
                FINAL_GRADE_COL, CYS_COL, ALGORITHM_COL,
                "Median_Grade_Calculated",
                "Max_Expert_Agreement_Count",
                "Majority_Grade_3plus",
            ]
        ].copy()

        algorithm_image_df["Algorithm_minus_expert_median"] = (
            algorithm_image_df[ALGORITHM_COL].astype(int)
            - algorithm_image_df["Median_Grade_Calculated"].astype(int)
        )
        algorithm_image_df["Absolute_grade_difference"] = (
            algorithm_image_df["Algorithm_minus_expert_median"].abs()
        )
        algorithm_image_df["Agreement_category"] = np.select(
            [
                algorithm_image_df["Algorithm_minus_expert_median"] == 0,
                algorithm_image_df["Algorithm_minus_expert_median"].abs() == 1,
                algorithm_image_df["Algorithm_minus_expert_median"].abs() >= 2,
            ],
            [
                "Exact match",
                "Within one grade",
                "Difference of two or more grades",
            ],
            default="Unclassified"
        )

        # Directional bias tests:
        # D = algorithm grade - expert median grade.
        differences = algo_grade - algo_reference
        lower_n = int(np.sum(differences < 0))
        equal_n = int(np.sum(differences == 0))
        higher_n = int(np.sum(differences > 0))
        discordant_n = lower_n + higher_n

        if discordant_n > 0:
            sign_p = float(
                binomtest(
                    higher_n,
                    n=discordant_n,
                    p=0.5,
                    alternative="two-sided"
                ).pvalue
            )
        else:
            sign_p = 1.0

        if np.all(differences == 0):
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        else:
            try:
                wres = wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto"
                )
                wilcoxon_stat = float(wres.statistic)
                wilcoxon_p = float(wres.pvalue)
            except ValueError:
                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan

        bias_test_df = pd.DataFrame([
            {"Statistic": "Algorithm lower than expert median (n)", "Value": lower_n},
            {"Statistic": "Equal to expert median (n)", "Value": equal_n},
            {"Statistic": "Algorithm higher than expert median (n)", "Value": higher_n},
            {"Statistic": "Discordant pairs (n)", "Value": discordant_n},
            {"Statistic": "Mean signed difference (algorithm - expert)", "Value": float(np.mean(differences))},
            {"Statistic": "Median signed difference (algorithm - expert)", "Value": float(np.median(differences))},
            {"Statistic": "Mean absolute grade difference", "Value": float(np.mean(np.abs(differences)))},
            {"Statistic": "Exact binomial sign-test P-value", "Value": sign_p},
            {"Statistic": "Wilcoxon signed-rank statistic", "Value": wilcoxon_stat},
            {"Statistic": "Wilcoxon signed-rank P-value", "Value": wilcoxon_p},
        ])

        print_section("6. CYS ALGORITHM VS FIVE-EXPERT MEDIAN")
        print(f"Paired images: {n_algorithm}")
        print(f"Exact agreement: {algo_result['Exact agreement n']}/{n_algorithm} "
              f"({algo_result['Exact agreement (%)']:.2f}%)")
        print(f"Within-one-grade agreement: "
              f"{algo_result['Within-one-grade agreement n']}/{n_algorithm} "
              f"({algo_result['Within-one-grade agreement (%)']:.2f}%)")
        print(f"Difference >=2 grades: {algo_result['Difference >=2 grades n']} "
              f"({algo_result['Difference >=2 grades (%)']:.2f}%)")
        print(f"Quadratically weighted kappa: "
              f"{algo_result['Quadratically weighted Cohen kappa']:.4f}")
        print(f"Spearman rho (CYS vs expert median): {rho:.4f}")
        print(f"Spearman P-value {p_text(p_spearman)}")
        print("\nConfusion matrix (rows = expert median; columns = CYS grade):")
        print(confusion_df.to_string())

        print("\nDirectional discrepancy:")
        print(f"Algorithm lower: {lower_n} ({lower_n / n_algorithm * 100:.2f}%)")
        print(f"Equal: {equal_n} ({equal_n / n_algorithm * 100:.2f}%)")
        print(f"Algorithm higher: {higher_n} ({higher_n / n_algorithm * 100:.2f}%)")
        print(f"Mean signed difference (algorithm - expert): {np.mean(differences):.4f}")
        print(f"Median signed difference: {np.median(differences):.4f}")
        print(f"Exact binomial sign-test P-value {p_text(sign_p)}")
        print(f"Wilcoxon signed-rank P-value {p_text(wilcoxon_p)}")

        # Sensitivity analysis: images with at least 3/5 experts assigning
        # the same grade. Majority grade is unique in this subset.
        sensitivity_mask = (
            algorithm_data["Max_Expert_Agreement_Count"].to_numpy() >= 3
        )
        sens_data = algorithm_data.loc[sensitivity_mask].copy()

        if len(sens_data) > 0:
            sens_reference = sens_data["Majority_Grade_3plus"].astype(int).to_numpy()
            sens_algorithm = sens_data[ALGORITHM_COL].astype(int).to_numpy()
            sens_result = agreement_metrics(sens_reference, sens_algorithm)

            sensitivity_metrics_df = pd.DataFrame([
                {"Metric": key, "Value": value}
                for key, value in sens_result.items()
            ])

            sens_cm = confusion_matrix(
                sens_reference, sens_algorithm, labels=GRADES
            )
            sensitivity_confusion_df = pd.DataFrame(
                sens_cm,
                index=[f"Majority {GRADE_NAMES[g]}" for g in GRADES],
                columns=[f"CYS {GRADE_NAMES[g]}" for g in GRADES]
            )
            sensitivity_confusion_df.index.name = "Majority reference grade"

            print_section("7. SENSITIVITY ANALYSIS: AT LEAST 3/5 EXPERTS AGREE")
            print(f"Images included: {len(sens_data)}")
            print(f"Exact agreement: {sens_result['Exact agreement n']}/"
                  f"{len(sens_data)} ({sens_result['Exact agreement (%)']:.2f}%)")
            print(f"Within-one-grade agreement: "
                  f"{sens_result['Within-one-grade agreement (%)']:.2f}%")
            print(f"Quadratically weighted kappa: "
                  f"{sens_result['Quadratically weighted Cohen kappa']:.4f}")
        else:
            print("\nNo images met the >=3/5 expert agreement criterion.")

    else:
        print_section("6. CYS ALGORITHM VS FIVE-EXPERT MEDIAN")
        print("No rows have both valid CYS values and valid Scientific_Grade values.")
        print("Expert reliability analyses were still completed using all expert-complete rows.")

    # ========================================================
    # 10. IMAGE-LEVEL EXPERT DATA
    # ========================================================

    image_level_df = expert_data[
        [IMAGE_COL] + EXPERT_COLS + [
            FINAL_GRADE_COL,
            "Median_Grade_Calculated",
            "Max_Expert_Agreement_Count",
            "Majority_Grade_3plus",
            CYS_COL,
            ALGORITHM_COL,
        ]
    ].copy()

    image_level_df["Final_Grade_matches_median"] = pd.Series(
        pd.NA, index=image_level_df.index, dtype="boolean"
    )
    _final_mask = image_level_df[FINAL_GRADE_COL].notna()
    image_level_df.loc[_final_mask, "Final_Grade_matches_median"] = (
        image_level_df.loc[_final_mask, FINAL_GRADE_COL].astype(int).to_numpy()
        == image_level_df.loc[_final_mask, "Median_Grade_Calculated"].astype(int).to_numpy()
    )

    # ========================================================
    # 11. DATA QUALITY SUMMARY
    # ========================================================

    data_quality_df = pd.DataFrame([
        {"Check": "Rows in original Excel", "Result": total_rows},
        {"Check": "Rows with empty Image_name", "Result": empty_name_count},
        {"Check": "Rows after removing empty Image_name", "Result": len(df)},
        {"Check": "Duplicate Image_name count", "Result": int(duplicate_mask.sum())},
        {"Check": "Images complete for all five experts", "Result": n_expert},
        {"Check": "Images complete for CYS and Scientific_Grade", "Result": n_algorithm},
        {"Check": "Expert-complete images missing Final_Grade", "Result":
            int(expert_data[FINAL_GRADE_COL].isna().sum())},
        {"Check": "Final_Grade vs calculated median mismatches", "Result": mismatch_count},
        {"Check": "Expert ratings used as primary reference", "Result":
            "Median recalculated from Expert 1-5"},
        {"Check": "Kendall's W interpretation", "Result":
            "Concordance in ranking of images across experts"},
        {"Check": "Pairwise kappa interpretation", "Result":
            "Chance-corrected ordinal agreement between two experts"},
    ])

    # ========================================================
    # 12. EXPORT MULTI-SHEET EXCEL WORKBOOK
    # ========================================================

    print_section("EXPORTING DETAILED RESULTS TO EXCEL")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        data_quality_df.to_excel(writer, sheet_name="Data_Check", index=False)
        kendall_df.to_excel(writer, sheet_name="Kendall_W", index=False)
        pairwise_df.to_excel(writer, sheet_name="Pairwise_Kappa", index=False)
        pairwise_summary_df.to_excel(writer, sheet_name="Pairwise_Summary", index=False)
        expert_grade_counts_df.to_excel(writer, sheet_name="Expert_Grade_Counts", index=False)
        agreement_dist_df.to_excel(writer, sheet_name="Agreement_Distribution", index=False)
        expert_vs_median_df.to_excel(writer, sheet_name="Expert_vs_Median", index=False)
        image_level_df.to_excel(writer, sheet_name="Image_Level_Expert_Data", index=False)
        mismatch_details.to_excel(writer, sheet_name="Median_Mismatch_Details", index=False)

        if n_algorithm > 0:
            algorithm_metrics_df.to_excel(writer, sheet_name="Algorithm_vs_Expert", index=False)
            confusion_df.to_excel(writer, sheet_name="Confusion_Matrix")
            algorithm_image_df.to_excel(writer, sheet_name="Algorithm_Image_Level", index=False)
            bias_test_df.to_excel(writer, sheet_name="Systematic_Bias_Tests", index=False)

            if not sensitivity_metrics_df.empty:
                sensitivity_metrics_df.to_excel(
                    writer, sheet_name="Sensitivity_3plus", index=False
                )
                sensitivity_confusion_df.to_excel(
                    writer, sheet_name="Sensitivity_Confusion"
                )

        if args.fisher_features is not None:
            overlap_stats.to_excel(writer, sheet_name="Fisher_Overlap", index=False)
            overlap_images.to_excel(writer, sheet_name="Fisher_Overlap_Images", index=False)

        # Short explanations for interpreting the sheets.
        guide_df = pd.DataFrame([
            ["Kendall_W", "Overall concordance in ranking the images across the five experts."],
            ["Pairwise_Kappa", "All ten expert pairs; quadratic weighted Cohen's kappa."],
            ["Pairwise_Summary", "Mean, SD, minimum and maximum of the ten pairwise kappas."],
            ["Expert_Grade_Counts", "Number and percentage of G1-G5 assigned by each expert."],
            ["Agreement_Distribution", "Number of images with maximum agreement of 5/5, 4/5, ..., 1/5."],
            ["Expert_vs_Median", "Descriptive only; each expert contributes to the five-expert median."],
            ["Algorithm_vs_Expert", "CYS-based grade compared with the median of five expert grades."],
            ["Confusion_Matrix", "Rows are expert median grades; columns are CYS algorithm grades."],
            ["Systematic_Bias_Tests", "Direction and magnitude of algorithm-minus-expert grade differences."],
            ["Sensitivity_3plus", "Algorithm comparison restricted to images with at least 3/5 expert agreement."],
            ["Median_Mismatch_Details", "Images where supplied Final_Grade differs from recalculated expert median."],
            ["Image_Level_Expert_Data", "Per-image expert ratings, median, maximum agreement and optional algorithm fields."],
            ["Algorithm_Image_Level", "Per-image CYS/algorithm comparison and grade discrepancy."],
        ], columns=["Worksheet", "Description"])
        if args.fisher_features is not None:
            guide_df.loc[len(guide_df)] = [
                "Fisher_Overlap", "Expert agreement by single-head image overlap with Fisher weighting."
            ]
            guide_df.loc[len(guide_df)] = [
                "Fisher_Overlap_Images", "Per-image comparison and Fisher cohort membership."
            ]
        guide_df.to_excel(writer, sheet_name="Read_Me", index=False)

        style_workbook(writer)

    print(f"Excel workbook saved to:\n{output_file}")

    print_section("FINAL SUMMARY")
    print(f"Expert-complete images: {n_expert}")
    print(f"Kendall's W: {W:.4f} (P {p_text(p_kendall)})")
    print(f"Mean pairwise quadratic weighted kappa: "
          f"{kappa_vals.mean():.4f} ± {kappa_vals.std(ddof=1):.4f}")
    print(f"Images with >=3/5 expert agreement: {at_least_3}/{n_expert} "
          f"({at_least_3 / n_expert * 100:.2f}%)")
    print(f"Final_Grade/median mismatches: {mismatch_count}")

    if n_algorithm > 0:
        print(f"Algorithm-comparison images: {n_algorithm}")
        print(f"Algorithm exact agreement: "
              f"{algo_result['Exact agreement (%)']:.2f}%")
        print(f"Algorithm within-one-grade agreement: "
              f"{algo_result['Within-one-grade agreement (%)']:.2f}%")
        print(f"Algorithm weighted kappa: "
              f"{algo_result['Quadratically weighted Cohen kappa']:.4f}")
        print(f"Spearman rho: {rho:.4f} (P {p_text(p_spearman)})")

    if args.print_image_details:
        print("\nPer-image details:")
        print(image_level_df.to_string(index=False))

    print("\nAnalysis completed successfully.")
    print(f"Output file:\n{output_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
