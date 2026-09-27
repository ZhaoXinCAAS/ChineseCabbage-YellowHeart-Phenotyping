"""GWAS interface: supports native Python engine (scikit-allel / NumPy) and PLINK binary."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from yheart.genetics.schemas import validate_gwas_table


@dataclass(frozen=True, slots=True)
class GwasArgs:
    """Parsed command-line arguments."""

    geno_prefix: str | None
    vcf: str | None
    pheno: str
    trait_col: str
    sample_col: str
    covar: str | None
    output: str
    method: str
    alpha: float
    max_variants: int | None
    min_maf: float


def parse_args(argv: Sequence[str] | None = None) -> GwasArgs:
    p = argparse.ArgumentParser(description="Run GWAS via Native Python Engine or PLINK binary.")
    p.add_argument(
        "--geno-prefix",
        type=str,
        default=None,
        help="PLINK binary prefix (.bed/.bim/.fam without extension).",
    )
    p.add_argument(
        "--vcf",
        type=str,
        default=None,
        help="VCF or VCF.GZ file path for native Python association testing.",
    )
    p.add_argument(
        "--pheno",
        type=str,
        required=True,
        help="Phenotype file (.csv or .xlsx) with sample IDs and trait column.",
    )
    p.add_argument(
        "--trait-col",
        type=str,
        default="CYS",
        help="Trait column name (default: CYS).",
    )
    p.add_argument(
        "--sample-col",
        type=str,
        default="sample_id",
        help="Sample identifier column in phenotype table (default: sample_id; fallback: QR).",
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
        choices=["assoc", "native", "plink"],
        help="Association method (default: assoc via plink; native for python engine).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for filtering (default: 0.05).",
    )
    p.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Maximum variants to test in native mode (useful for testing/fast scans).",
    )
    p.add_argument(
        "--min-maf",
        type=float,
        default=0.05,
        help="Minor allele frequency threshold for native testing (default: 0.05).",
    )
    args = p.parse_args(argv)
    return cast(GwasArgs, args)


def run_gwas_native(
    vcf_path: str,
    pheno_path: str,
    trait_col: str = "CYS",
    sample_col: str = "sample_id",
    output_path: str = "./gwas_results.csv",
    alpha: float = 0.05,
    max_variants: int | None = None,
    min_maf: float = 0.05,
) -> pd.DataFrame:
    """Perform native single-variant ordinary least squares linear regression via scikit-allel."""
    import allel

    if not Path(vcf_path).is_file():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    if not Path(pheno_path).is_file():
        raise FileNotFoundError(f"Phenotype file not found: {pheno_path}")

    if pheno_path.lower().endswith(".xlsx"):
        raw_pheno = pd.read_excel(pheno_path)
    else:
        raw_pheno = pd.read_csv(pheno_path)

    id_col = sample_col if sample_col in raw_pheno.columns else ("QR" if "QR" in raw_pheno.columns else None)
    if not id_col:
        raise KeyError(f"Phenotype table missing sample column (looked for '{sample_col}', 'QR').")
    if trait_col not in raw_pheno.columns:
        raise KeyError(f"Phenotype table missing trait column '{trait_col}'.")

    # Aggregate by sample if duplicates exist (e.g. multi-replicate mean)
    valid_pheno = raw_pheno[[id_col, trait_col]].dropna()
    pheno_series = valid_pheno.groupby(id_col)[trait_col].mean()

    # Pre-read samples from VCF to establish matching indices
    sample_header = allel.read_vcf_headers(vcf_path).samples
    vcf_samples = [str(s) for s in sample_header]

    # Map phenotype values to VCF samples; support year prefix matching if exact match not found
    exact_map = {str(k): float(v) for k, v in pheno_series.items()}
    # Fallback suffix map (e.g. 24CC0440 -> CC0440, matching 22CC0440)
    suffix_map = {str(k)[4:]: float(v) for k, v in exact_map.items() if len(str(k)) > 4}

    matched_indices: list[int] = []
    y_values: list[float] = []

    for idx, s in enumerate(vcf_samples):
        if s in exact_map:
            matched_indices.append(idx)
            y_values.append(exact_map[s])
        elif len(s) > 4 and s[4:] in suffix_map:
            matched_indices.append(idx)
            y_values.append(suffix_map[s[4:]])

    if len(matched_indices) < 3:
        raise ValueError(
            f"Fewer than 3 overlapping samples found between VCF ({len(vcf_samples)} samples) "
            f"and Phenotype ({len(exact_map)} samples)."
        )

    print(f"Native GWAS: matched {len(matched_indices)} phenotype samples with VCF genotype samples.")
    y = np.array(y_values, dtype=np.float64)
    # Center phenotype
    y_centered = y - np.mean(y)
    ss_y = float(np.sum(y_centered**2))
    n_samples = len(y)
    deg_freedom = n_samples - 2

    # Read VCF chunks and compute linear regression per variant
    fields = ["variants/CHROM", "variants/POS", "variants/ID", "variants/REF", "variants/ALT", "calldata/GT"]
    results: list[dict[str, object]] = []

    tested_count = 0
    # Use chunked iterator for memory-efficient streaming across massive VCFs
    chunk_size = 5000
    _, _, _, chunk_iter = allel.iter_vcf_chunks(vcf_path, fields=fields, chunk_length=chunk_size, numbers={"calldata/GT": 2})
    for chunk_tuple in chunk_iter:
        chunk = chunk_tuple[0]  # Dictionary of arrays
        chroms = chunk["variants/CHROM"]
        positions = chunk["variants/POS"]
        var_ids = chunk["variants/ID"]
        refs = chunk["variants/REF"]
        alts = chunk["variants/ALT"]
        # shape: (n_variants, n_vcf_samples, 2)
        gt_raw = chunk["calldata/GT"][:, matched_indices, :]
        gt = allel.GenotypeArray(gt_raw)
        dosages = gt.to_n_alt(fill=-1).astype(np.float64)  # 0, 1, 2, or -1 (missing)

        n_variants = len(positions)
        for i in range(n_variants):
            x = dosages[i]
            valid_mask = x >= 0
            n_valid = int(np.sum(valid_mask))
            if n_valid < 10:
                continue

            x_v = x[valid_mask]
            # Check MAF
            allele_freq = float(np.mean(x_v)) / 2.0
            maf = min(allele_freq, 1.0 - allele_freq)
            if maf < min_maf:
                continue

            y_v = y_centered[valid_mask]
            x_centered = x_v - np.mean(x_v)
            ss_x = float(np.sum(x_centered**2))
            if ss_x < 1e-12:
                continue

            s_xy = float(np.sum(x_centered * y_v))
            beta = s_xy / ss_x
            ss_res = max(0.0, float(np.sum(y_v**2)) - beta * s_xy)
            se = np.sqrt(ss_res / (max(1, n_valid - 2) * ss_x))
            if se <= 0 or np.isnan(se):
                continue

            t_stat = beta / se
            p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=n_valid - 2))
            if np.isnan(p_val):
                continue

            var_id = str(var_ids[i])
            if not var_id or var_id == ".":
                var_id = f"{chroms[i]}_{positions[i]}"

            alt_allele = alts[i][0] if isinstance(alts[i], (list, np.ndarray)) else str(alts[i])
            results.append({
                "CHR": str(chroms[i]),
                "SNP": var_id,
                "BP": int(positions[i]),
                "A1": str(alt_allele),
                "BETA": float(beta),
                "SE": float(se),
                "P": float(p_val),
            })
            tested_count += 1
            if max_variants and tested_count >= max_variants:
                break

        if max_variants and tested_count >= max_variants:
            break

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("Warning: No variants satisfied quality/MAF filters.")
        empty_res = pd.DataFrame(columns=["CHR", "SNP", "BP", "A1", "BETA", "SE", "P"])
        empty_res.to_csv(output_path, index=False)
        return empty_res

    # Sort by chromosome and position
    res_df = res_df.sort_values(by=["CHR", "BP"]).reset_index(drop=True)
    if alpha < 1.0:
        sig_df = res_df[res_df["P"] < alpha]
    else:
        sig_df = res_df

    sig_df.to_csv(output_path, index=False)
    print(f"Native GWAS complete. Tested: {len(res_df)} SNPs, Significant (P < {alpha}): {len(sig_df)}")
    print(f"Results saved to: {output_path}")
    return res_df


def run_gwas_plink(args: GwasArgs) -> None:
    """Run GWAS using the system PLINK binary."""
    plink_check = subprocess.run(
        ["plink", "--help"],
        capture_output=True,
        check=False,
    )
    if plink_check.returncode != 0:
        print("plink binary not found in PATH; install PLINK or add to PATH.")
        sys.exit(1)

    if not args.geno_prefix:
        print("Error: --geno-prefix is required for plink method.")
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

    sample_col = args.sample_col if args.sample_col in pheno_df.columns else "sample_id"
    required_cols = {sample_col, args.trait_col}
    if missing := required_cols - set(pheno_df.columns):
        raise KeyError(f"Phenotype file missing required columns: {missing}")

    # Prepare standard PLINK phenotype table (FID, IID, Trait)
    with tempfile.NamedTemporaryFile(suffix=".pheno", delete=False, mode="w") as tmp_f:
        tmp_pheno_path = tmp_f.name
        sub_df = pheno_df[[sample_col, args.trait_col]].copy()
        sub_df.insert(0, "FID", sub_df[sample_col])
        sub_df.rename(columns={sample_col: "IID"}, inplace=True)
        sub_df.to_csv(tmp_f, sep="\t", index=False)

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
    assoc_df = pd.read_csv(assoc_path, sep=r"\s+", dtype={"SNP": str, "CHR": str})
    assoc_df.columns = [c.strip() for c in assoc_df.columns]

    if "P" in assoc_df.columns:
        sig_df = assoc_df[assoc_df["P"] < args.alpha]
    else:
        sig_df = assoc_df

    sig_df.to_csv(args.output, index=False)
    print(f"GWAS complete. Results saved to: {args.output}")
    print(f"SNPs tested: {len(assoc_df)}, Significant (P < {args.alpha}): {len(sig_df)}")

    for p in (
        tmp_pheno_path,
        assoc_path,
        tmp_prefix + ".log",
        tmp_prefix + ".nosex",
    ):
        with contextlib.suppress(OSError):
            Path(p).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None):
    """Entry point for GWAS console script."""
    args = parse_args(argv)

    if args.method == "native" or (args.vcf and args.method != "assoc" and args.method != "plink"):
        vcf_path = args.vcf or (f"{args.geno_prefix}.vcf.gz" if args.geno_prefix else None)
        if not vcf_path:
            # Check default location in workspace
            default_vcf = Path("GWASData/SNPs/AA_AABB.snp.clean.rapa.maf0.05.recode.vcf.gz")
            if default_vcf.is_file():
                vcf_path = str(default_vcf)
            else:
                print("Error: Native GWAS requires --vcf <path_to_vcf_file>.")
                sys.exit(1)
        run_gwas_native(
            vcf_path=vcf_path,
            pheno_path=args.pheno,
            trait_col=args.trait_col,
            sample_col=args.sample_col,
            output_path=args.output,
            alpha=args.alpha,
            max_variants=args.max_variants,
            min_maf=args.min_maf,
        )
    else:
        run_gwas_plink(args)


if __name__ == "__main__":
    main(sys.argv[1:])
