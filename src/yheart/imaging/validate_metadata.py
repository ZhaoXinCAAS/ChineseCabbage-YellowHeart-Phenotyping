"""Sample-sheet metadata validation (Reviewer-2 Q1).

R2-1 asks whether harvest timing was uniform across genotypes and how long
samples waited between harvest and imaging. Code cannot run the field trial,
but it can *enforce the provenance*: this script requires a sample sheet with
`sample_id / genotype / harvest_time / imaging_time` columns, parses the
timestamps, audits the harvest→imaging interval per sample, and flags missing
values, unparsable timestamps, and intervals beyond `--max-interval-h`.

Usage
-----
.. code-block:: bash
    uv run validate-metadata --input ./data/sample_sheet.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd

DEFAULT_INPUT = "./samples_images/sample_sheet.csv"
DEFAULT_OUTPUT = "./samples_images/results/metadata_audit.csv"
DEFAULT_MAX_INTERVAL_H = 48.0

REQUIRED_COLS = ("sample_id", "genotype", "harvest_time", "imaging_time")


class AuditSummary(TypedDict):
    """Per-sheet audit counters."""

    n: int
    n_flagged: int
    median_interval_h: float | None
    genotypes: int


def read_table(path: str):
    """Read a sample-sheet CSV/XLSX file."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format (need .csv/.xlsx): {path}")


def audit(df: pd.DataFrame, max_interval_h: float = DEFAULT_MAX_INTERVAL_H):
    """Check required columns and flag bad or over-long harvest-to-imaging intervals."""
    if missing := [c for c in REQUIRED_COLS if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    assert isinstance(out, pd.DataFrame)
    out["harvest_time"] = pd.to_datetime(out["harvest_time"], errors="coerce")
    out["imaging_time"] = pd.to_datetime(out["imaging_time"], errors="coerce")
    valid = out["harvest_time"].notna() & out["imaging_time"].notna()
    out["interval_h"] = float("nan")
    delta = out.loc[valid, "imaging_time"] - out.loc[valid, "harvest_time"]
    assert isinstance(delta, pd.Series)
    out.loc[valid, "interval_h"] = delta.apply(
        lambda td: td.total_seconds() / 3600 if isinstance(td, pd.Timedelta) else float("nan")
    )
    out["flag"] = ""
    bad_time = out[["harvest_time", "imaging_time"]].isna().any(axis=1)
    out.loc[bad_time, "flag"] = "missing/bad-timestamp"
    over = out["interval_h"].notna() & (out["interval_h"] > max_interval_h)
    out.loc[over, "flag"] = out.loc[over, "flag"].apply(
        lambda s: (s + "; " if s else "") + "long-interval"
    )
    neg = out["interval_h"].notna() & (out["interval_h"] < 0)
    out.loc[neg, "flag"] = out.loc[neg, "flag"].apply(
        lambda s: (s + "; " if s else "") + "negative-interval"
    )
    intervals = out["interval_h"].dropna()
    assert isinstance(intervals, pd.Series)
    summary: AuditSummary = {
        "n": len(out),
        "n_flagged": len(out[out["flag"] != ""]),
        "median_interval_h": None if intervals.empty else round(intervals.median(), 2),
        "genotypes": len(out.groupby("genotype").size()),
    }
    return out, summary


@dataclass(frozen=True, slots=True)
class ValidateMetadataArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    max_interval_h: float


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Audit harvest→imaging metadata.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--max-interval-h", type=float, default=DEFAULT_MAX_INTERVAL_H)
    return cast(ValidateMetadataArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    try:
        df = read_table(args.input)
        table, summary = audit(df, args.max_interval_h)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(
        f"n={summary['n']}, genotypes={summary['genotypes']}, "
        f"flagged={summary['n_flagged']}, "
        f"median_interval_h={summary['median_interval_h']}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
