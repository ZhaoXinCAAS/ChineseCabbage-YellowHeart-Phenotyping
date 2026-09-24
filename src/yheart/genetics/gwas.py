"""GWAS interface via PLINK binary (system binary required)."""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd


@dataclass(frozen=True, slots=True)
class GwasArgs:
    """Parsed command-line arguments."""

    geno_prefix: str
    pheno: str
    trait_col: str
    covar: str | None
    output: str
    method: str
    alpha: float


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Run GWAS via PLINK binary.")
    p.add_argument(
        "--geno-prefix",
        type=str,
        required=True,
        help="PLINK binary prefix (.bed/.bim/.fam without extension).",
    )
    p.add_argument(
        "--pheno",
        type=str,
        required=True,
        help="Phenotype file (.csv or .xlsx) with columns sample_id and trait.",
    )
    p.add_argument(
        "--trait-col",
        type=str,
        default="CYS",
        help="Trait column name (default: CYS).",
    )
    p.add_argument(
        "--covar",
        type=str,
        default=None,
        help="Covariate file (.txt/.csv), standard PLINK format (optional).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./gwas_results.csv",
        help="Output association results CSV.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="assoc",
        choices=["assoc"],
        help="Association method (default: assoc).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for filtering (default: 0.05).",
    )
    return cast(GwasArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    """Run the gwas console script via the PLINK binary."""
    args = parse_args(argv)

    plink_check = subprocess.run(
        ["plink", "--help"],
        capture_output=True,
        check=False,
    )
    if plink_check.returncode != 0:
        print("plink binary not found in PATH; install PLINK or add to PATH.")
        sys.exit(1)

    for ext in (".bed", ".bim", ".fam"):
        path = Path(f"{args.geno_prefix}{ext}")
        if not path.is_file():
            print(f"Genotype file not found: {path}")
            sys.exit(1)

    if not Path(args.pheno).is_file():
        raise FileNotFoundError(f"Phenotype file not found: {args.pheno}")

    if args.pheno.lower().endswith(".xlsx"):
        pheno_df: pd.DataFrame = pd.read_excel(args.pheno)
    else:
        pheno_df = pd.read_csv(args.pheno)

    if pheno_df.empty:
        raise ValueError("Phenotype file is empty.")

    required_cols = {"sample_id", args.trait_col}
    if missing := required_cols - set(pheno_df.columns):
        raise KeyError(f"Phenotype file missing required columns: {missing}")
    with tempfile.NamedTemporaryFile(suffix=".pheno", delete=False, mode="w") as tmp_f:
        tmp_pheno_path = tmp_f.name

    tmp_fd, tmp_prefix = tempfile.mkstemp(prefix="gwas_tmp_")
    os.close(tmp_fd)
    cmd = [
        "plink",
        "--bfile",
        args.geno_prefix,
        "--pheno",
        tmp_pheno_path,
        "--pheno-name",
        args.trait_col,
        "--assoc",
        "--out",
        tmp_prefix,
    ]
    if args.covar:
        cmd += ["--covar", args.covar]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"plink execution failed (exit {result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    assoc_path = tmp_prefix + ".assoc"

    # Identifier columns read as str at the source; numeric columns are
    # inferred and contract-checked downstream (GwasColumns).
    assoc_df = pd.read_csv(assoc_path, sep=r"\s+", dtype={"SNP": str, "CHR": str})

    assoc_df.columns = [c.strip() for c in assoc_df.columns]

    if "P" in assoc_df.columns:
        sig_df = assoc_df[assoc_df["P"] < args.alpha]
    else:
        sig_df = assoc_df

    sig_df.to_csv(args.output, index=False)
    print(f"GWAS complete. Results saved to: {args.output}")
    print(f"SNPs tested: {len(assoc_df)}, Significant (P < {args.alpha}): {len(sig_df)}")
    if len(sig_df) > 0 and "P" in sig_df.columns:
        min_p_row: pd.Series = sig_df.nsmallest(1, columns="P").iloc[0]
        print(f"Top SNP: {min_p_row.get('SNP', 'N/A')} (P={min_p_row.get('P', 'N/A')})")

    for p in (
        tmp_pheno_path,
        assoc_path,
        tmp_prefix + ".log",
        tmp_prefix + ".nosex",
    ):
        with contextlib.suppress(OSError):
            Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1:])
