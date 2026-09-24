"""Tests for the heritability pseudoreplication guard (Reviewer-1 Q4)."""

import numpy as np
import pandas as pd
import pytest


from yheart.genetics.heritability import (
    aggregate_technical_replicates,
    compute_h2_table,
    compute_h2_with_lmm,
    fit_single_h2,
)


def _heritable_df(
    seed: int = 0,
    n_geno: int = 12,
    n_rep: int = 5,
    between: float = 2.0,
    within: float = 0.2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = rng.normal(0, between, n_geno)
    rows: list[dict[str, object]] = []
    for i, m in enumerate(means):
        rows.extend(
            {
                "QR": f"G{i}",
                "plant_id": f"G{i}-P{j}",
                "CYS": m + rng.normal(0, within),
            }
            for j in range(n_rep)
        )
    return pd.DataFrame(rows)


def test_strong_genetic_signal_gives_high_h2():
    df = _heritable_df()
    fit = fit_single_h2(df, "QR", "CYS")
    assert fit is not None
    assert fit["H2"] > 0.8
    assert fit["n_genotypes"] == 12 and fit["n_obs"] == 60
    assert fit["n_h"] == 5.0


def test_no_genetic_signal_gives_low_h2():
    # Pure noise independent of genotype labels: no heritable signal.
    df = _heritable_df(seed=1, n_geno=30, n_rep=8, between=0.0, within=1.0)
    fit = fit_single_h2(df, "QR", "CYS")
    assert fit is not None
    assert fit["H2"] < 0.3


def test_technical_replicates_aggregate_to_plants():
    df = _heritable_df()
    # Duplicate every image (two technical shots per plant).
    dup = pd.concat([df, df.assign(CYS=df["CYS"] + 1e-6)], ignore_index=True)
    agg = aggregate_technical_replicates(dup, "QR", "CYS", "plant_id")
    assert agg is not None
    assert len(agg) == len(df)  # one row per plant again
    table = compute_h2_table(
        dup,
        "QR",
        {"CYS": "CYS"},
        levels=("image", "aggregate"),
        replicate_col="plant_id",
    )
    assert set(table["level"]) == {"image", "aggregate"}
    agg_row = table.loc[table["level"] == "aggregate"].iloc[0]
    assert agg_row["n_obs"] == len(df) < len(dup)


def test_aggregate_without_replicate_col_is_skipped(
    capsys: pytest.CaptureFixture[str],
):
    df = _heritable_df()
    table = compute_h2_table(
        df,
        "QR",
        {"CYS": "CYS"},
        levels=("aggregate",),
        replicate_col=None,
    )
    assert table.empty
    assert "replicate-col" in capsys.readouterr().out


def test_legacy_wrapper_still_returns_plain_dict():
    df = _heritable_df()
    out = compute_h2_with_lmm(df, "QR", {"CYS": "CYS"})
    assert isinstance(out["CYS"], float) and out["CYS"] > 0.8
