"""Unit tests for the gene annotation CLI module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yheart.genetics import annotate_genes


def test_parse_args_defaults() -> None:
    args = annotate_genes.parse_args([
        "--gwas", "gwas.csv",
        "--annotation", "genes.gtf",
    ])
    assert args.gwas == "gwas.csv"
    assert args.annotation == "genes.gtf"
    assert args.sig_threshold == 5e-8
    assert args.window_bp == 500000
    assert args.output == "./annotated_genes.csv"


def test_parse_annotation_valid(tmp_path: Path) -> None:
    gtf_file = tmp_path / "test.gtf"
    gtf_file.write_text(
        "# comment header\n"
        "chr1\t1000\t2000\tBra01\n"
        "chr2\t5000\t6000\tBra02\n",
        encoding="utf-8",
    )
    df = annotate_genes.parse_annotation(str(gtf_file))
    assert len(df) == 2
    assert list(df.columns) == ["chr", "start", "end", "gene"]
    assert df.iloc[0]["gene"] == "Bra01"
    assert df.iloc[0]["start"] == 1000


def test_parse_annotation_empty_raises(tmp_path: Path) -> None:
    empty_gtf = tmp_path / "empty.gtf"
    empty_gtf.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="yielded no valid gene rows"):
        annotate_genes.parse_annotation(str(empty_gtf))


def test_main_exits_when_files_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        annotate_genes.main(["--gwas", "missing.csv", "--annotation", "fake.gtf"])
    assert exc_info.value.code == 1


def test_main_exits_on_empty_gwas(tmp_path: Path) -> None:
    gwas_file = tmp_path / "empty_gwas.csv"
    gwas_file.touch()
    annot_file = tmp_path / "genes.gtf"
    annot_file.write_text("chr1\t1000\t2000\tBra01\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        annotate_genes.main(["--gwas", str(gwas_file), "--annotation", str(annot_file)])
    assert exc_info.value.code == 1


def test_main_handles_no_significant_snps(tmp_path: Path) -> None:
    gwas_file = tmp_path / "gwas.csv"
    pd.DataFrame({
        "SNP": ["s1"],
        "CHR": ["chr1"],
        "BP": [1500],
        "P": [0.5],
    }).to_csv(gwas_file, index=False)

    annot_file = tmp_path / "genes.gtf"
    annot_file.write_text("chr1\t1000\t2000\tBra01\n", encoding="utf-8")

    out_file = tmp_path / "annotated.csv"
    with pytest.raises(SystemExit) as exc_info:
        annotate_genes.main([
            "--gwas", str(gwas_file),
            "--annotation", str(annot_file),
            "--output", str(out_file),
            "--sig-threshold", "5e-8",
        ])
    assert exc_info.value.code == 0
    assert out_file.is_file()
    res = pd.read_csv(out_file)
    assert len(res) == 0
    assert "nearest_gene" in res.columns


def test_main_annotates_significant_snps(tmp_path: Path) -> None:
    gwas_file = tmp_path / "gwas.csv"
    pd.DataFrame({
        "SNP": ["s1", "s2"],
        "CHR": ["chr1", "chr1"],
        "BP": [1200, 50000],
        "P": [1e-9, 1e-10],
    }).to_csv(gwas_file, index=False)

    annot_file = tmp_path / "genes.gtf"
    annot_file.write_text(
        "chr1\t1000\t2000\tBra01001\n"
        "chr1\t1800\t2500\tBra01002\n"
        "chr2\t1000\t2000\tBra02001\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "annotated.csv"
    annotate_genes.main([
        "--gwas", str(gwas_file),
        "--annotation", str(annot_file),
        "--output", str(out_file),
        "--sig-threshold", "1e-5",
        "--window-bp", "10000",
    ])

    assert out_file.is_file()
    res = pd.read_csv(out_file)
    assert len(res) == 2

    # First SNP at 1200 is close to Bra01001 (1000-2000)
    s1_row = res[res["SNP"] == "s1"].iloc[0]
    assert s1_row["nearest_gene"] == "Bra01001"
    assert "Bra01001" in s1_row["genes_in_window"]
    assert "Bra01002" in s1_row["genes_in_window"]

    # Second SNP at 50000 has no genes in window of 10000 bp
    s2_row = res[res["SNP"] == "s2"].iloc[0]
    assert pd.isna(s2_row["nearest_gene"]) or s2_row["nearest_gene"] == ""
