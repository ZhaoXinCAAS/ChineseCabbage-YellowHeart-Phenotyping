"""Gene annotation for significant GWAS SNPs using minimal GTF/GFF format."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
from pandera.errors import SchemaError

from yheart.genetics.schemas import gwas_col, validate_gwas_table


@dataclass(frozen=True, slots=True)
class AnnotateGenesArgs:
    """Parsed command-line arguments."""

    gwas: str
    annotation: str
    sig_threshold: float
    window_bp: int
    output: str


class GeneRecord(TypedDict):
    """One parsed annotation row (chr/start/end/gene)."""

    chr: str
    start: int
    end: int
    gene: str


def parse_args(argv: Sequence[str] | None = None) -> AnnotateGenesArgs:
    p = argparse.ArgumentParser(description="Annotate significant GWAS SNPs with nearby genes.")
    p.add_argument(
        "--gwas",
        type=str,
        required=True,
        help="GWAS results CSV (SNP, CHR, BP, P).",
    )
    p.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="Gene annotation file (GTF/GFF-like: tab-delimited with chr, start, end, gene).",
    )
    p.add_argument(
        "--sig-threshold",
        type=float,
        default=5e-8,
        help="Significance threshold (default: 5e-8).",
    )
    p.add_argument(
        "--window-bp",
        type=int,
        default=500000,
        help="Flanking window in bp (default: 500000).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./annotated_genes.csv",
        help="Output annotated genes CSV.",
    )
    return cast(AnnotateGenesArgs, p.parse_args(argv))


def parse_annotation(path: str):
    """Parse a minimal GTF/GFF-like annotation file into a DataFrame."""
    rows: list[GeneRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            rows.append({
                "chr": parts[0],
                "start": int(parts[1]),
                "end": int(parts[2]),
                "gene": parts[3],
            })
        except ValueError:
            continue
    if not rows:
        raise ValueError(
            f"Annotation file {path} yielded no valid gene rows. "
            "Expected format: tab-delimited with at least 4 columns (chr, start, end, gene)."
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    if not Path(args.gwas).is_file():
        print(f"GWAS file not found: {args.gwas}")
        sys.exit(1)

    if not Path(args.annotation).is_file():
        print(f"Annotation file not found: {args.annotation}")
        print(
            "Expected minimal GTF/GFF format: tab-delimited lines "
            "(chr start end gene ...). Skip header lines starting with #."
        )
        sys.exit(1)

    gwas_df = pd.read_csv(args.gwas, dtype={"SNP": str, "CHR": str})
    if gwas_df.empty:
        print("GWAS file is empty.")
        sys.exit(1)

    try:
        gdf = validate_gwas_table(gwas_df)
    except SchemaError as e:
        print(f"GWAS CSV failed validation: {e}")
        sys.exit(1)

    sig_df = gdf[gwas_col(gdf, "P") < args.sig_threshold].copy()
    if sig_df.empty:
        print("No significant SNPs to annotate.")
        out_df = pd.DataFrame(
            columns=[
                "SNP",
                "CHR",
                "BP",
                "P",
                "nearest_gene",
                "distance_to_gene_start",
                "genes_in_window",
            ]
        )
        out_df.to_csv(args.output, index=False)
        print(f"Output file (header only) saved to: {args.output}")
        sys.exit(0)

    try:
        annot_df = parse_annotation(args.annotation)
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    results: list[dict[str, object]] = []
    for _, snp_row in sig_df.iterrows():
        snp_chr = str(snp_row["CHR"])
        snp_bp = cast(int, snp_row["BP"])
        window_start = max(0, snp_bp - args.window_bp)
        window_end = snp_bp + args.window_bp

        same_chr = annot_df[annot_df["chr"] == snp_chr]
        assert isinstance(same_chr, pd.DataFrame)
        overlapping = same_chr[(same_chr["start"] < window_end) & (same_chr["end"] > window_start)]
        assert isinstance(overlapping, pd.DataFrame)

        genes_list: list[str] = (
            [] if overlapping.empty else overlapping["gene"].astype(str).tolist()
        )

        nearest_gene = ""
        distance: int | str = ""
        if not overlapping.empty:
            distances: list[tuple[int, str, int]] = []
            for record in overlapping.to_dict("records"):
                start, end = int(record["start"]), int(record["end"])
                dist = min(abs(snp_bp - start), abs(snp_bp - end))
                distances.append((dist, str(record["gene"]), start))
            distances.sort(key=lambda x: x[0])
            nearest_gene = distances[0][1]
            distance = distances[0][0]
        else:
            nearest_gene = ";".join(genes_list) if genes_list else ""

        results.append({
            "SNP": snp_row["SNP"],
            "CHR": snp_row["CHR"],
            "BP": snp_row["BP"],
            "P": snp_row["P"],
            "nearest_gene": nearest_gene,
            "distance_to_gene_start": distance,
            "genes_in_window": ",".join(genes_list),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(
        f"Annotated {len(out_df)} significant SNPs. Genes in window (window: {args.window_bp} bp)."
    )
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
