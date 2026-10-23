"""Tests for CYS_compute (Reviewer-1 Q2 ablation)."""

import numpy as np
import pandas as pd
from pathlib import Path


from yheart.scoring.cys_compute import (
    build_cys_table,
    compute_cys,
    compute_yellow_score,
    load_fisher_weights,
)


def _write(tmp_path: Path, df: pd.DataFrame, name: str) -> str:
    p = tmp_path / name
    df.to_excel(p, index=False)
    return str(p)


def test_multiply_penalizes_small_deep_vs_additive(tmp_path: Path):
    norm = pd.DataFrame({
        "filename": ["small_deep.jpg", "large_pale.jpg"],
        "Yellow_Ratio": [0.10, 0.80],
        "b_mean": [0.95, 0.40],
        "S_mean": [0.90, 0.35],
    })
    fisher = pd.DataFrame({
        "Feature_Name": ["b_mean", "S_mean"],
        "Fisher_Score": [1.0, 1.0],
        "Response_Direction": ["Positive (+)", "Positive (+)"],
        "Final_Weight": [0.5, 0.5],
    })
    npath = _write(tmp_path, norm, "norm.xlsx")
    fpath = _write(tmp_path, fisher, "fisher.xlsx")
    tab = build_cys_table(npath, fpath, top_k=2, mode="all")
    # small-deep: multiply << additive (over-penalization diagnostic)
    sm = tab.loc[tab["filename"] == "small_deep.jpg"].iloc[0]
    assert sm["CYS"] < sm["CYS_additive"]
    assert bool(sm["flag_small_deep"])
    # geometric sits between multiply and additive
    assert sm["CYS"] <= sm["CYS_geometric"] <= sm["CYS_additive"]
    # Product and geometric scores preserve ordering; additive can differ.
    lg = tab.loc[tab["filename"] == "large_pale.jpg"].iloc[0]
    assert (sm["CYS"] > lg["CYS"]) == (sm["CYS_geometric"] > lg["CYS_geometric"])


def test_negative_direction_uses_complement(tmp_path: Path):
    norm = pd.DataFrame({
        "filename": ["a.jpg"],
        "Yellow_Ratio": [0.5],
        "B_mean": [0.2],  # low blue => very yellow
    })
    fisher = pd.DataFrame({
        "Feature_Name": ["B_mean"],
        "Fisher_Score": [1.2],
        "Response_Direction": ["Negative (-)"],
        "Final_Weight": [-1.0],
    })
    _write(tmp_path, norm, "norm.xlsx")
    fpath = _write(tmp_path, fisher, "fisher.xlsx")
    w = load_fisher_weights(fpath, top_k=1)
    assert w["B_mean"] == (1.0, -1)
    score = compute_yellow_score(norm, w)
    assert abs(float(score.iloc[0]) - 0.8) < 1e-9  # 1 - 0.2
    cys = compute_cys(norm["Yellow_Ratio"], score, "multiply")
    assert abs(float(cys.iloc[0]) - 0.4) < 1e-9


def test_mojibake_headers_fallback_by_position(tmp_path: Path):
    # Simulates shipped fisher_weights_418.xlsx with garbled Chinese headers.
    df = pd.DataFrame([
        ["ExR_mean", 1.6, "x", 0.25],
        ["S_mean", 1.4, "x", 0.22],
    ])
    p = tmp_path / "garbled.xlsx"
    df.to_excel(p, index=False, header=["\ufffd\ufffd", "a", "b", "c"])
    w = load_fisher_weights(str(p), top_k=2)
    assert set(w) == {"ExR_mean", "S_mean"}
    assert abs(sum(v[0] for v in w.values()) - 1.0) < 1e-9


def test_missing_measurements_stay_missing_and_geometric_zero_is_exact():
    ratio = pd.Series([0.0, float("nan"), 0.5])
    score = pd.Series([0.8, 0.8, float("nan")])
    for mode in ("multiply", "additive", "geometric", "harmonic"):
        result = compute_cys(ratio, score, mode)
        assert result.iloc[1:].isna().all()
    assert compute_cys(ratio, score, "geometric").iloc[0] == 0
    assert compute_cys(ratio, score, "harmonic").iloc[0] == 0


def test_harmonic_mean_formula():
    ratio = pd.Series([0.5, 0.2, 0.9])
    score = pd.Series([0.5, 0.8, 0.1])
    result = compute_cys(ratio, score, "harmonic")
    expected = 2 * ratio * score / (ratio + score)
    assert np.allclose(result.to_numpy(), expected.to_numpy())
