"""Unit tests for the Manhattan plot CLI module."""

from __future__ import annotations

from pathlib import Path

import holoviews as hv
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
        "--input",
        str(gwas_csv),
        "--output",
        str(out_png),
        "--title",
        "Test Manhattan",
    ])

    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_lead_label_frame_handles_empty_and_close_snps() -> None:
    assert manhattan_plot.lead_label_frame([]) is None

    snps = [
        pd.Series({"SNP": "rs1", "x": 50_000_000, "-log10P": 8.0, "CHR": "1", "BP": 500}),
        pd.Series({"SNP": "rs2", "x": 52_000_000, "-log10P": 8.1, "CHR": "1", "BP": 520}),
        pd.Series({"SNP": "rs3", "x": 200_000_000, "-log10P": 7.5, "CHR": "2", "BP": 300}),
    ]
    frame = manhattan_plot.lead_label_frame(snps, total_x_len=300_000_000, y_max=10.0)
    assert frame is not None
    assert len(frame) == 3
    assert "x_label" in frame.columns
    assert "y_label" in frame.columns
    # Close SNPs rs1 and rs2 adjust horizontally into clearance space
    x_label = frame["x_label"].to_numpy(dtype=float)
    x = frame["x"].to_numpy(dtype=float)
    assert x_label[0] < x[0]
    assert x_label[1] > x[1]
    # Isolated rs3 stays centered
    assert x_label[2] == x[2]


def test_main_generates_interactive_html_successfully(tmp_path: Path) -> None:
    gwas_csv = tmp_path / "gwas_valid_html.csv"
    pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(1, 11)],
        "CHR": ["chr1"] * 5 + ["chr2"] * 5,
        "BP": [100 * i for i in range(1, 6)] + [100 * i for i in range(1, 6)],
        "P": [1e-9, 1e-6, 0.01, 0.05, 0.5, 1e-8, 1e-4, 0.02, 0.1, 0.8],
    }).to_csv(gwas_csv, index=False)

    out_html = tmp_path / "out_manhattan.html"
    manhattan_plot.main([
        "--input",
        str(gwas_csv),
        "--output",
        str(out_html),
        "--title",
        "Test Interactive Manhattan",
    ])

    assert out_html.is_file()
    assert out_html.stat().st_size > 0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--sig-threshold", "0"),
        ("--sig-threshold", "nan"),
        ("--sig-threshold", "1.1"),
        ("--suggestive-threshold", "-1"),
        ("--suggestive-threshold", "inf"),
    ],
)
def test_rejects_invalid_thresholds(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        manhattan_plot.parse_args(["--input", "unused.csv", flag, value])
    assert exc_info.value.code == 2


def test_args_are_real_dataclass() -> None:
    args = manhattan_plot.parse_args(["--input", "unused.csv", "--suggestive-threshold", "0"])
    assert isinstance(args, manhattan_plot.ManhattanArgs)
    assert args.suggestive_threshold == 0


def test_lead_selection_distance_boundary_and_chromosomes() -> None:
    frame = pd.DataFrame({
        "SNP": ["best", "near", "boundary", "other"],
        "CHR": ["1", "1", "1", "2"],
        "BP": [0, 999_999, 1_000_000, 0],
        "P": [1e-10, 1e-9, 1e-8, 1e-7],
    })
    leads = manhattan_plot._select_lead_snps(frame)
    assert [lead["SNP"] for lead in leads] == ["best", "boundary", "other"]
    assert len(manhattan_plot._select_lead_snps(frame, limit=2)) == 2


@pytest.mark.parametrize("extension", ["png", "html"])
def test_zero_position_and_zero_p_render(tmp_path: Path, extension: str) -> None:
    source = tmp_path / "zero.csv"
    pd.DataFrame({"SNP": ["zero"], "CHR": ["1"], "BP": [0], "P": [0.0]}).to_csv(source, index=False)
    output = tmp_path / f"zero.{extension}"
    manhattan_plot.main(["--input", str(source), "--output", str(output)])
    assert output.stat().st_size > 0


def test_large_coordinates_and_original_zero_p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large.csv"
    pd.DataFrame({
        "SNP": ["large", "zero"],
        "CHR": ["1", "2"],
        "BP": [3_000_000_000, 1],
        "P": [0.0, 0.5],
    }).to_csv(source, index=False)
    captured: list[object] = []

    def capture(layout: object, *args: object, **kwargs: object) -> None:
        captured.append(layout)

    monkeypatch.setattr(hv, "save", capture)
    manhattan_plot.main(["--input", str(source)])
    layout = captured[0]
    assert isinstance(layout, hv.Overlay)
    points = list(layout.values())[0]
    assert points.data["x"].tolist() == [3_000_000_000, 3_000_000_001]
    assert points.data["P"].tolist() == [0.0, 0.5]
