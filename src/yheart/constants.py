"""Canonical trait vocabulary (Reviewer-2 Q3).

Single source of truth for column names, paper abbreviations, and plot
labels. File-format columns stay verbose (`Yellow_Ratio`); the paper-facing
symbols are `YR` (yellow-heart area ratio) and `YS` (yellow score):

* `YR`  = `Yellow_Ratio`  (R2-3: was "YR" in some figures)
* `YS`  = `Yellow_score`  (R2-3: was "YS" in some figures)
* `CYS` = comprehensive score (`Yellow_Ratio × Yellow_score` by default)

`FEATURES` is the canonical 10-trait order; `fisher_scores.FEATURES` and
`normalize_minmax.COLOR_FEATURES` must match it as sets (guarded by tests).
"""

from __future__ import annotations

RATIO_COL = "Yellow_Ratio"
SCORE_COL = "Yellow_score"
CYS_COL = "CYS"
GENOTYPE_COL = "QR"

FEATURES: tuple[str, ...] = (
    "H_circular_mean_deg",
    "S_mean",
    "V_mean",
    "L_mean",
    "a_mean",
    "b_mean",
    "R_mean",
    "G_mean",
    "B_mean",
    "ExR_mean",
)

# column -> (abbreviation, LaTeX label)
SYMBOLS = {
    "Yellow_Ratio": ("YR", "Yellow Ratio"),
    "Yellow_score": ("YS", "Yellow Score"),
    "CYS": ("CYS", "CYS"),
    "H_circular_mean_deg": ("H", r"$\mathrm{H}$"),
    "S_mean": ("S", r"$\mathrm{S}$"),
    "V_mean": ("V", r"$\mathrm{V}$"),
    "L_mean": ("L", r"$\mathrm{L}$"),
    "a_mean": ("a*", r"$\mathrm{a^*}$"),
    "b_mean": ("b*", r"$\mathrm{b^*}$"),
    "R_mean": ("R", r"$\mathrm{R}$"),
    "G_mean": ("G", r"$\mathrm{G}$"),
    "B_mean": ("B", r"$\mathrm{B}$"),
    "ExR_mean": ("ExR", r"$\mathrm{ExR}$"),
}


def abbr(column: str):
    """Paper abbreviation for a trait column (falls back to the column)."""
    return SYMBOLS.get(column, (column, column))[0]
