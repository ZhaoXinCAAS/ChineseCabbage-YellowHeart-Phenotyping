"""Unit tests for the GWAS CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from yheart.genetics import gwas


def test_parse_args_defaults() -> None:
    args = gwas.parse_args([
        "--geno-prefix", "sample_prefix",
        "--pheno", "traits.csv",
    ])
    assert args.geno_prefix == "sample_prefix"
    assert args.pheno == "traits.csv"
    assert args.trait_col == "CYS"
    assert args.covar is None
    assert args.output == "./gwas_results.csv"
    assert args.method == "assoc"
    assert args.alpha == 0.05


def test_main_exits_when_plink_not_found(tmp_path: Path) -> None:
    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(SystemExit) as exc_info,
    ):
        mock_run.return_value = MagicMock(returncode=127)
        gwas.main(["--geno-prefix", "fake", "--pheno", "fake.csv"])
    assert exc_info.value.code == 1


def test_main_exits_when_geno_missing(tmp_path: Path) -> None:
    # PLINK is available
    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(SystemExit) as exc_info,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        gwas.main([
            "--geno-prefix", str(tmp_path / "missing_prefix"),
            "--pheno", "fake.csv",
        ])
    assert exc_info.value.code == 1


def test_main_raises_when_pheno_missing(tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    for ext in (".bed", ".bim", ".fam"):
        (tmp_path / f"geno{ext}").touch()

    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(FileNotFoundError, match="Phenotype file not found"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        gwas.main([
            "--geno-prefix", str(prefix),
            "--pheno", str(tmp_path / "missing_pheno.csv"),
        ])


def test_main_raises_when_pheno_missing_required_columns(tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    for ext in (".bed", ".bim", ".fam"):
        (tmp_path / f"geno{ext}").touch()

    pheno_file = tmp_path / "pheno.csv"
    pd.DataFrame({"wrong_id": ["S1"], "other_col": [1.2]}).to_csv(pheno_file, index=False)

    with (
        patch("subprocess.run") as mock_run,
        pytest.raises(KeyError, match="Phenotype file missing required columns"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        gwas.main([
            "--geno-prefix", str(prefix),
            "--pheno", str(pheno_file),
        ])


def test_main_end_to_end_mocked(tmp_path: Path) -> None:
    prefix = tmp_path / "geno"
    for ext in (".bed", ".bim", ".fam"):
        (tmp_path / f"geno{ext}").touch()

    pheno_file = tmp_path / "pheno.csv"
    pd.DataFrame({
        "sample_id": ["S1", "S2", "S3"],
        "CYS": [0.2, 0.5, 0.8],
    }).to_csv(pheno_file, index=False)

    out_file = tmp_path / "gwas_out.csv"

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[0] == "plink" and "--help" in cmd:
            return MagicMock(returncode=0)
        if cmd[0] == "plink" and "--assoc" in cmd:
            out_idx = cmd.index("--out")
            tmp_pfx = cmd[out_idx + 1]
            assoc_content = (
                " CHR         SNP         BP   A1       F_A       F_U   A2        CHISQ            P           OR \n"
                "   1        rs01        100    A       0.5       0.2    G        10.5      0.001          2.5 \n"
                "   1        rs02        200    T       0.1       0.1    C         0.1      0.800          1.0 \n"
            )
            Path(tmp_pfx + ".assoc").write_text(assoc_content, encoding="utf-8")
            return MagicMock(returncode=0)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        gwas.main([
            "--geno-prefix", str(prefix),
            "--pheno", str(pheno_file),
            "--output", str(out_file),
            "--alpha", "0.05",
        ])

    assert out_file.is_file()
    res = pd.read_csv(out_file)
    assert len(res) == 1
    assert res.iloc[0]["SNP"] == "rs01"
    assert res.iloc[0]["P"] == 0.001
