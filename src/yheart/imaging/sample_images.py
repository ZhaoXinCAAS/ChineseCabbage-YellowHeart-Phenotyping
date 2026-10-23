r"""Build a deterministic stratified image subset.

Selects up to ``--per-folder`` images from every class or grade subfolder,
copies them into one flat folder with a group prefix (so names cannot
collide), and writes a manifest of source -> output paths. This reproduces
the 30-images-per-grade sampling used for the illumination-robustness
dataset, with a fixed seed for reproducible subsets.

Usage
-----
.. code-block:: bash
    uv run sample-images --input ./data/Graded_Images \
        --output ./data/Robustness_Graded_Images_Dataset
    uv run sample-images --input ./data/Graded_Images --per-folder 20 \
        --groups Grade_2,Grade_3 --seed 7
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from yheart.imaging.extract_color_features import VALID_EXTS

DEFAULT_PER_FOLDER = 30
DEFAULT_SEED = 42
MANIFEST_NAME = "sampling_manifest.csv"


def group_files(
    input_dir: str | Path, groups: Sequence[str] | None = None
) -> dict[str, list[Path]]:
    """Return ``{group: sorted image files}`` for the requested subfolders."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    names = list(groups) if groups else sorted(p.name for p in root.iterdir() if p.is_dir())
    missing = [name for name in names if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"Group folders not found: {missing}")
    return {
        name: sorted(
            p for p in (root / name).iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
        )
        for name in names
    }


def sample_tree(
    input_dir: str,
    output_dir: str,
    per_folder: int = DEFAULT_PER_FOLDER,
    seed: int = DEFAULT_SEED,
    groups: Sequence[str] | None = None,
    prefix: bool = True,
) -> pd.DataFrame:
    """Copy a deterministic subset and return the manifest rows."""
    if per_folder < 1:
        raise ValueError("per_folder must be at least 1")
    root = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Output directory must be outside the input directory")
    destination.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for group, files in group_files(root, groups).items():
        count = min(per_folder, len(files))
        if count == 0:
            continue
        chosen: np.ndarray = np.sort(rng.choice(len(files), size=count, replace=False))
        for index in chosen:
            source = files[int(index)]
            name = f"{group}_{source.name}" if prefix else source.name
            target = destination / name
            shutil.copy2(source, target)
            rows.append({
                "group": group,
                "filename": source.name,
                "source": str(source),
                "output": str(target),
            })
    return pd.DataFrame(rows, columns=["group", "filename", "source", "output"])


@dataclass(frozen=True, slots=True)
class SampleImagesArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    per_folder: int
    seed: int
    groups: str | None
    no_prefix: bool
    manifest: str | None


def parse_args(argv: Sequence[str] | None = None) -> SampleImagesArgs:
    p = argparse.ArgumentParser(description="Sample a stratified image subset.")
    p.add_argument("--input", required=True, help="Root folder of class/grade subfolders.")
    p.add_argument("--output", required=True, help="Destination for the copied subset.")
    p.add_argument("--per-folder", type=int, default=DEFAULT_PER_FOLDER)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--groups", default=None, help="Comma-separated subfolders (default: all).")
    p.add_argument("--no-prefix", action="store_true", help="Keep original file names.")
    p.add_argument("--manifest", default=None, help="Manifest CSV path.")
    return cast(SampleImagesArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    groups = [s.strip() for s in args.groups.split(",") if s.strip()] if args.groups else None
    try:
        manifest = sample_tree(
            args.input,
            args.output,
            args.per_folder,
            args.seed,
            groups,
            prefix=not args.no_prefix,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output) / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    if manifest.empty:
        print("No images copied (check subfolders and image extensions).")
        sys.exit(1)
    print(manifest.groupby("group").size().to_string())
    print(f"Copied {len(manifest)} image(s) to {args.output}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
