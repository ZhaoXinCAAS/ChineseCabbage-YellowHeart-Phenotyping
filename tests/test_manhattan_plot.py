"""Unit tests for the Manhattan plot CLI module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yheart.genetics import manhattan_plot


def test_parse_args_defaults() -> None:
    args = manhattan_plot.parse_args(["--input", "results.csv"])
    assert args.input == "results.csv"
    assert args.output == "./manhattan_plot.png"
    assert args.chr_label == "Chromosome"
    assert args.sig_threshold == 5e-8
    assert args.suggestive_threshold == 1e-5
    assert args.title == "Manhattan Plot"


def test_main_exits_on_empty_csv(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.touch()
    with pytest.raises(SystemExit) as exc_info:
        manhattan_plot.main(["--input", str(empty_csv)])
    assert exc_info.value.code == 1


def test_main_exits_on_schema_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"SNP": ["rs1"], "CHR": ["1"]}).to_csv(bad_csv, index=False)
    with pytest.raises(SystemExit) as exc_info:
        manhattan_plot.main(["--input", str(bad_csv)])
    assert exc_info.value.code == 1


def test_main_exits_when_no_valid_p_values(tmp_path: Path) -> None:
    invalid_p_csv = tmp_path / "invalid_p.csv"
    pd.DataFrame({
        "SNP": ["rs1"],
        "CHR": ["1"],
        "BP": [100],
        "P": [-0.5],
    }).to_csv(invalid_p_csv, index=False)
    with pytest.raises(SystemExit) as exc_info:
        manhattan_plot.main(["--input", str(invalid_p_csv)])
    assert exc_info.value.code == 1


def test_main_generates_plot_successfully(tmp_path: Path) -> None:
    gwas_csv = tmp_path / "gwas_valid.csv"
    pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(1, 11)],
        "CHR": ["chr1"] * 5 + ["chr2"] * 5,
        "BP": [100 * i for i in range(1, 6)] + [100 * i for i in range(1, 6)],
        "P": [1e-9, 1e-6, 0.01, 0.05, 0.5, 1e-8, 1e-4, 0.02, 0.1, 0.8],
    }).to_csv(gwas_csv, index=False)

    out_png = tmp_path / "out_manhattan.png"
    manhattan_plot.main([
        "--input", str(gwas_csv),
        "--output", str(out_png),
        "--title", "Test Manhattan",
    ])

    assert out_png.is_file()
    assert out_png.stat().st_size > 0
