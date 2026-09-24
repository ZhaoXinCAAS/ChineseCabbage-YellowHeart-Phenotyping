"""Tests for validate_thresholds (Reviewer-1 Q3/Q7)."""

from pathlib import Path
import pandas as pd
import pytest


from yheart.scoring.validate_thresholds import run


def _graded(tmp_path: Path, rows: list[dict[str, str | int | float]]):
    p = tmp_path / "graded.xlsx"
    pd.DataFrame(rows).to_excel(p, index=False)
    return str(p)


def _ref(tmp_path: Path, rows: list[dict[str, str | int | float]], name: str = "ref.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_perfect_expert_agreement(tmp_path: Path):
    rows = [
        {"filename": f"s{i}.jpg", "Scientific_Grade": (i % 5) + 1, "CYS": 0.1 * i}
        for i in range(20)
    ]
    g = _graded(tmp_path, rows)
    r = _ref(
        tmp_path, [{"filename": x["filename"], "expert_grade": x["Scientific_Grade"]} for x in rows]
    )
    assert (
        run(
            g,
            r,
            "filename",
            "Scientific_Grade",
            "expert_grade",
            "CYS",
            "carotenoid",
            str(tmp_path / "m.csv"),
            str(tmp_path / "c.csv"),
        )
        is True
    )
    m = pd.read_csv(tmp_path / "m.csv")
    assert m["accuracy_vs_expert"].iloc[0] == 1.0
    assert m["quadratic_kappa_vs_expert"].iloc[0] == 1.0
    c = pd.read_csv(tmp_path / "c.csv", index_col=0)
    assert c.to_numpy().trace() == 20


def test_hplc_monotonic_correlation(tmp_path: Path):
    rows = [
        {"filename": f"s{i}.jpg", "Scientific_Grade": (i % 5) + 1, "CYS": round(0.05 * i, 3)}
        for i in range(10)
    ]
    g = _graded(tmp_path, rows)
    r = _ref(
        tmp_path,
        [{"filename": x["filename"], "carotenoid": float(x["CYS"]) * 100 + 5} for x in rows],
    )
    assert (
        run(
            g,
            r,
            "filename",
            "Scientific_Grade",
            "expert_grade",
            "CYS",
            "carotenoid",
            str(tmp_path / "m.csv"),
            str(tmp_path / "c.csv"),
        )
        is True
    )
    m = pd.read_csv(tmp_path / "m.csv")
    assert m["hplc_spearman_rho"].iloc[0] == 1.0
    assert m["hplc_pearson_r"].iloc[0] > 0.99


def test_missing_reference_columns_skip_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    g = _graded(tmp_path, [{"filename": "a.jpg", "Scientific_Grade": 1, "CYS": 0.1}])
    r = _ref(tmp_path, [{"filename": "a.jpg", "other": 5}])
    assert (
        run(
            g,
            r,
            "filename",
            "Scientific_Grade",
            "expert_grade",
            "CYS",
            "carotenoid",
            str(tmp_path / "m.csv"),
            str(tmp_path / "c.csv"),
        )
        is False
    )
    out = capsys.readouterr().out
    assert "skipped" in out
    assert not (tmp_path / "m.csv").exists()
