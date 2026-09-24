r"""Comprehensive Yellow Score (CYS) synthesis with formulation ablation.

Reconstructs the missing link ``normalized + Fisher weights -> CYS`` that
``gmm_grading.py`` / ``heritability.py`` consume, and answers Reviewer-1 Q2 (does the
multiplicative form over-penalize "small-area but deep-yellow" samples?).
Formulations (all inputs clipped to [0, 1]; negative-direction features
use the complement ``1 - x`` so larger always means "more yellow"):
* ``multiply`` (paper default): ``CYS = Yellow_Ratio * Yellow_score``
* ``additive``: ``CYS = alpha * Yellow_Ratio + (1 - alpha) * Yellow_score``
* ``geometric``: ``CYS = sqrt(Yellow_Ratio * Yellow_score)``
``Yellow_score = sum_k w_k * t_k`` where ``w_k`` are the renormalized
absolute Fisher weights of the top-k features and ``t_k`` the
direction-aligned normalized traits.

Usage
-----
.. code-block:: bash
    uv run cys-compute  # demo defaults
    uv run cys-compute --normalized ./data/normalized_data_1319.xlsx \
        --fisher ./data/fisher_weights_418.xlsx --output ./data/CYS_1319.xlsx
    uv run cys-compute --mode all --top-k 4  # ablation for R1-2
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

DEFAULT_NORMALIZED = "./samples_images/results/normalized_data_1319.xlsx"
DEFAULT_FISHER = "./samples_images/results/fisher_weights_418.xlsx"
DEFAULT_OUTPUT = "./samples_images/results/CYS_1319.xlsx"
DEFAULT_TOP_K = 4
DEFAULT_ALPHA = 0.5
EPS = 1e-9
RATIO_COL = "Yellow_Ratio"


def _pick_col(df: pd.DataFrame, candidates: Sequence[str], pos: int) -> str:
    """Select a named column, falling back to a stable positional column.

    The source workbook may contain mojibake Chinese headers, so position is
    the only stable fallback key.
    """
    return next((c for c in candidates if c in df.columns), df.columns[pos])


def load_fisher_weights(path: str, top_k: int = DEFAULT_TOP_K) -> dict[str, tuple[float, int]]:
    """Return `{feature: (renorm_abs_weight, sign)}` for top-k features.

    Robust to mojibake headers: falls back to column positions
    `0=name, 1=score, 3=Final_Weight`.
    """
    df = pd.read_excel(path)
    name_col = _pick_col(df, ["Feature_Name", "Feature", "feature"], 0)
    weight_col = _pick_col(df, ["Final_Weight", "Weight", "weight", "FinalWeight"], 3)
    direction_col = _pick_col(df, ["Response_Direction", "Direction", "direction"], 2)
    names = df[name_col].astype(str).str.strip()

    # Prefer signed Final_Weight; fall back to direction string.
    signs: list[int] = []
    weights: list[float] = []
    for _, row in df.iterrows():
        w = cast(float, row[weight_col])
        weights.append(w)
        d = str(row[direction_col])
        if "Nega" in d or "-" in d or "负" in d or "��" in d and w < 0:
            signs.append(-1)
        elif "Posi" in d or "+" in d or "正" in d:
            signs.append(1)
        else:
            signs.append(1 if w >= 0 else -1)
    order: np.ndarray = np.argsort([abs(w) for w in weights])[::-1][:top_k]
    selected: list[tuple[str, float, int]] = [
        (str(names.iloc[int(i)]), weights[int(i)], signs[int(i)]) for i in order
    ]
    total = sum(abs(w) for _, w, _ in selected) or 1.0
    return {n: (abs(w) / total, s) for n, w, s in selected}


def aligned_traits(df: pd.DataFrame, weights: dict[str, tuple[float, int]]) -> pd.DataFrame:
    """Direction-aligned traits in [0, 1]; negative features use ``1 - x``."""
    out = pd.DataFrame(index=df.index)
    for feat, (_, sign) in weights.items():
        if feat not in df.columns:
            raise KeyError(f"Normalized table missing feature column: {feat}")

        # Quoted: pandas Series is not subscriptable at runtime.
        x: "pd.Series[float]" = df[feat].clip(0, 1)
        out[feat] = x if sign >= 0 else 1.0 - x
    return out


def compute_yellow_score(df_norm: pd.DataFrame, weights: dict[str, tuple[float, int]]) -> pd.Series:
    """Weighted sum of direction-aligned normalized traits."""
    t = aligned_traits(df_norm, weights)
    w = np.array([weights[f][0] for f in t.columns], dtype=float)
    return pd.Series(t.to_numpy() @ w, index=df_norm.index, name="Yellow_score")


def compute_cys(
    ratio: pd.Series, score: pd.Series, mode: str, alpha: float = DEFAULT_ALPHA
) -> pd.Series:
    """Combine the area ratio and the yellow score (multiply/additive/geometric)."""
    r: "pd.Series[float]" = ratio.clip(0, 1).fillna(0.0)
    s: "pd.Series[float]" = score.clip(0, 1).fillna(0.0)
    match mode:
        case "additive":
            result = alpha * r + (1 - alpha) * s
        case "geometric":
            result = pd.Series(
                np.sqrt(np.clip(r.to_numpy(), EPS, 1) * np.clip(s.to_numpy(), EPS, 1)),
                index=r.index,
            )
        case "multiply":
            result: pd.Series = r * s
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return result.rename("CYS")


def build_cys_table(
    normalized_path: str,
    fisher_path: str,
    top_k: int = DEFAULT_TOP_K,
    mode: str = "all",
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Assemble the full CYS table including ablation columns."""
    df = pd.read_excel(normalized_path)
    if RATIO_COL not in df.columns:
        raise KeyError(f"Normalized table missing {RATIO_COL}")
    weights = load_fisher_weights(fisher_path, top_k=top_k)
    score = compute_yellow_score(df, weights)
    ratio: "pd.Series[float]" = df[RATIO_COL].clip(0, 1)
    out = df[["filename"]].copy() if "filename" in df.columns else pd.DataFrame()
    assert isinstance(out, pd.DataFrame)
    out[RATIO_COL] = ratio
    for feat in weights:
        out[feat] = df[feat].clip(0, 1)
        if weights[feat][1] < 0:
            out[f"1-{feat.replace('_mean', '')}"] = 1.0 - out[feat]
    for feat, (w, _) in weights.items():
        out[f"{feat.split('_')[0]}_weight"] = round(w, 4)

    # Keep genotype passthrough for heritability.py (column 'QR' when present).
    if "QR" in df.columns:
        out["QR"] = df["QR"]
    out["Yellow_score"] = score.round(6)
    modes = ["multiply", "additive", "geometric"] if mode == "all" else [mode]
    for m in modes:
        col = (
            "CYS"
            if (mode != "all" and m == mode) or (mode == "all" and m == "multiply")
            else f"CYS_{m}"
        )
        out[col] = compute_cys(ratio, score, m, alpha).round(6)
    if mode == "all":
        # R1-2 diagnostic: small-area but deep-yellow candidates.
        out["flag_small_deep"] = (ratio < 0.2) & (score > 0.7)
    return out


@dataclass(frozen=True, slots=True)
class CysComputeArgs:
    """Parsed command-line arguments."""

    normalized: str
    fisher: str
    output: str
    top_k: int
    mode: str
    alpha: float


def parse_args(argv: Sequence[str] | None = None) -> CysComputeArgs:
    p = argparse.ArgumentParser(description="Synthesize CYS with ablation.")
    p.add_argument("--normalized", default=DEFAULT_NORMALIZED)
    p.add_argument("--fisher", default=DEFAULT_FISHER)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--mode",
        choices=["multiply", "additive", "geometric", "all"],
        default="all",
    )
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return cast(CysComputeArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    try:
        table = build_cys_table(
            args.normalized,
            args.fisher,
            top_k=args.top_k,
            mode=args.mode,
            alpha=args.alpha,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(args.output, index=False)
    n_flag = cast(int, table["flag_small_deep"].sum()) if "flag_small_deep" in table else 0
    print(
        f"CYS table ({args.mode}) saved to: {args.output} "
        f"[n={len(table)}, small-deep flags={n_flag}]"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
