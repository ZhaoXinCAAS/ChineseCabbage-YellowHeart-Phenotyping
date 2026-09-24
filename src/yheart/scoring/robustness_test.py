"""Illumination robustness test (Reviewer-1 Q6).

Re-runs the two critical pipeline stages — yellow-heart segmentation
(``extract_yellow_contour``) and color-trait extraction
(``process_image_pair``) — on brightness / color-temperature perturbed copies
of sampled images, plus a gray-world white-balanced variant, and reports:

* segmentation stability: IoU(baseline mask, perturbed mask) per level;
* feature stability: mean |Δ| of ``Yellow_Ratio, S/b/ExR/B_mean`` with an
  explicit ``B_mean`` row (blue-channel negative-weight discussion, R1-6);
* scaler ablation (``--scaler-compare``): the same sampled traits scaled by
  MinMax vs. RobustScaler (fit on baseline, applied to all) — mean
  cross-perturbation std per trait, smaller means a more stable normalization.

Usage
-----
.. code-block:: bash
    uv run robustness-test --input ./samples_images/results/annotations_head_region
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from yheart.imaging.color_calibrate import (
    apply_brightness,
    apply_temp_shift,
    gray_world_wb,
)
from yheart.imaging.extract_color_features import (
    VALID_EXTS,
    process_image_pair,
)
from yheart.imaging.segment_yellow_heart import (
    extract_yellow_contour,
    load_json_and_get_mask,
)

DEFAULT_INPUT = "./samples_images/results/Auto_segment_Yellow_heart/json"
DEFAULT_OUTPUT = "./samples_images/results/robustness_summary.csv"
DEFAULT_SCALER_OUTPUT = "./samples_images/results/scaler_stability.csv"

# Drift columns d_<trait> / mean_d_<trait> are mirrored in PerturbRecord/PerturbSummary below.
KEY_TRAITS = ["Yellow_Ratio", "S_mean", "b_mean", "ExR_mean", "B_mean"]
BRIGHTNESS_LEVELS = (0.8, 1.0, 1.2)
TEMP_LEVELS = (0.9, 1.0, 1.1)


class PerturbSummary(TypedDict):
    """One perturbation-level summary row (mean_d_* keys mirror KEY_TRAITS)."""

    perturb: str
    n: int
    mean_seg_iou: float
    fail_rate: float
    mean_d_Yellow_Ratio: float
    mean_d_S_mean: float
    mean_d_b_mean: float
    mean_d_ExR_mean: float
    mean_d_B_mean: float


class PerturbRecord(TypedDict):
    """One perturbation record from ``evaluate_image`` (``d_*`` keys mirror KEY_TRAITS).

    The `d_<trait>` drift keys stay `NotRequired`: the `KEY_TRAITS` loop
    writes them through a plain-dict view after the literal below (TypedDict
    rejects computed-key subscripts), so they cannot be present at construction.
    """

    filename: str
    perturb: str
    seg_iou: float
    extract_ok: bool
    d_Yellow_Ratio: NotRequired[float]
    d_S_mean: NotRequired[float]
    d_b_mean: NotRequired[float]
    d_ExR_mean: NotRequired[float]
    d_B_mean: NotRequired[float]

    # Raw per-trait feature dicts consumed by scaler_ablation.
    _base: dict[str, float]
    _pert: dict[str, float] | None


def mask_iou(a: np.ndarray, b: np.ndarray):
    """IoU of two binary masks (empty-empty scores 1)."""
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1) if a.any() or b.any() else np.float64(1)


def baseline_and_perturbed_masks(img: np.ndarray, head_mask: np.ndarray):
    """Baseline yellow mask used as the reference for perturbations."""
    _, base_bin = extract_yellow_contour(img, head_mask)
    return (base_bin > 0).astype(np.uint8)


def evaluate_image(img_path: Path, levels: Sequence[str]) -> list[PerturbRecord]:
    """Perturb one image; return one record per perturbation level."""
    img = cv2.imread(str(img_path))
    json_path = img_path.parent / f"{img_path.stem}.json"
    if img is None or not json_path.exists():
        return []
    head_mask = load_json_and_get_mask(str(json_path), img.shape)
    base_bin = baseline_and_perturbed_masks(img, head_mask)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t_img = Path(tmp) / img_path.name
            t_json = Path(tmp) / f"{img_path.stem}.json"
            cv2.imwrite(str(t_img), img)
            shutil.copy(str(json_path), str(t_json))
            base_feat = process_image_pair(str(t_img), str(t_json))
    except ValueError:
        return []  # baseline itself unextractable: nothing to compare

    variants: dict[str, np.ndarray] = {"identity": img}
    for f in BRIGHTNESS_LEVELS:
        variants[f"brightness@{f}"] = apply_brightness(img, f)
    for g in TEMP_LEVELS:
        variants[f"temp@{g}"] = apply_temp_shift(img, g)
    variants["wb-grayworld"] = gray_world_wb(img)

    records: list[PerturbRecord] = []
    for name in levels:
        pert = variants[name]
        _, pert_bin = extract_yellow_contour(pert, head_mask)
        iou = mask_iou(base_bin > 0, pert_bin > 0)
        feat: dict[str, str | int | float] | None = None
        with contextlib.suppress(ValueError):
            with tempfile.TemporaryDirectory() as tmp:
                t_img = Path(tmp) / img_path.name
                t_json = Path(tmp) / f"{img_path.stem}.json"
                cv2.imwrite(str(t_img), pert)
                shutil.copy(str(json_path), str(t_json))
                feat = process_image_pair(str(t_img), str(t_json))
        rec: PerturbRecord = {
            "filename": img_path.name,
            "perturb": name,
            "seg_iou": round(iou, 4),
            "extract_ok": feat is not None,
            "_base": {t: cast(float, base_feat[t]) for t in KEY_TRAITS if t in base_feat},
            "_pert": (
                {t: cast(float, feat[t]) for t in KEY_TRAITS if t in feat}
                if feat is not None
                else None
            ),
        }

        # Computed d_<trait> keys can't be subscripted on a TypedDict; write
        # them through a plain-dict view (declared NotRequired on PerturbRecord).
        drift_cols = cast(dict[str, object], rec)
        for t in KEY_TRAITS:
            drift_cols[f"d_{t}"] = (
                round(abs(cast(float, feat[t]) - cast(float, base_feat[t])), 6)
                if feat is not None and t in feat
                else float("nan")
            )
        records.append(rec)
    return records


def _mean_drift(g: pd.DataFrame, trait: str) -> float:
    """Mean absolute trait drift over one perturbation group (NaN-safe)."""
    # Quoted: pandas Series is not subscriptable at runtime.
    delta: "pd.Series[float]" = g[f"d_{trait}"]
    return round(delta.mean(skipna=True), 6)


def summarize(records: list[PerturbRecord]) -> pd.DataFrame:
    """Mean segmentation IoU and feature drift per perturbation level."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in records])
    rows: list[PerturbSummary] = []
    rows.extend(
        {
            "perturb": str(perturb),
            "n": len(g),
            "mean_seg_iou": round(g["seg_iou"].mean(), 4),
            "fail_rate": round(1.0 - g["extract_ok"].mean(), 4),
            "mean_d_Yellow_Ratio": _mean_drift(g, "Yellow_Ratio"),
            "mean_d_S_mean": _mean_drift(g, "S_mean"),
            "mean_d_b_mean": _mean_drift(g, "b_mean"),
            "mean_d_ExR_mean": _mean_drift(g, "ExR_mean"),
            "mean_d_B_mean": _mean_drift(g, "B_mean"),
        }
        for perturb, g in df.groupby("perturb")
    )
    return pd.DataFrame(rows)


def scaler_ablation(records: list[PerturbRecord]) -> pd.DataFrame:
    """Fit MinMax/Robust scalers on baseline rows; compare spread over perts."""
    base_rows: list[dict[str, float]] = []
    pert_rows: list[dict[str, object]] = []
    for r in records:
        base = r["_base"]
        pert = r["_pert"]
        if r["perturb"] == "identity":
            base_rows.append({t: base[t] for t in KEY_TRAITS if t in base})
        if pert is not None:
            pert_rows.append({
                "filename": r["filename"],
                **{t: pert[t] for t in KEY_TRAITS if t in pert},
            })
    base = pd.DataFrame(base_rows)
    pert = pd.DataFrame(pert_rows)
    if base.empty or pert.empty:
        return pd.DataFrame()
    out: list[dict[str, object]] = []
    scalers: tuple[tuple[MinMaxScaler | RobustScaler, str], ...] = (
        (MinMaxScaler(), "minmax"),
        (RobustScaler(), "robust"),
    )
    for scaler, name in scalers:
        scaler.fit(base[KEY_TRAITS].to_numpy(dtype=float))
        stds: dict[str, np.ndarray] = {}
        for fn, g in pert.groupby("filename"):
            transformed: np.ndarray = np.asarray(
                scaler.transform(g[KEY_TRAITS].to_numpy(dtype=float)), dtype=float
            )
            stds[str(fn)] = np.std(transformed, axis=0)
        arr = np.array(list(stds.values()))
        out.extend(
            {
                "trait": trait,
                "scaler": name,
                "mean_cross_pert_std": round(arr[:, j].mean(), 6),
            }
            for j, trait in enumerate(KEY_TRAITS)
        )
    return pd.DataFrame(out)


def run_robustness(
    input_dir: str, sample: int, seed: int, scaler_compare: bool = True
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Sample images, perturb them, and summarize stability."""
    files = sorted([
        p
        for p in Path(input_dir).iterdir()
        if p.suffix.lower() in VALID_EXTS and (p.parent / f"{p.stem}.json").exists()
    ])
    if not files:
        print(f"Error: no images with JSON sidecars in '{input_dir}'.")
        return None, None
    rng = np.random.default_rng(seed)
    chosen: np.ndarray = np.asarray(
        rng.choice(len(files), size=min(sample, len(files)), replace=False)
    )
    picked: list[Path] = [files[int(i)] for i in chosen]
    levels = (
        ["identity"]
        + [f"brightness@{f}" for f in BRIGHTNESS_LEVELS]
        + [f"temp@{g}" for g in TEMP_LEVELS]
        + ["wb-grayworld"]
    )
    records: list[PerturbRecord] = []
    for img_path in picked:
        records.extend(evaluate_image(img_path, levels))
    if not records:
        print("Warning: baseline extraction failed on all sampled images.")
        return None, None
    summary = summarize(records)
    scaler_df = scaler_ablation(records) if scaler_compare else pd.DataFrame()
    return summary, scaler_df


@dataclass(frozen=True, slots=True)
class RobustnessArgs:
    """Parsed command-line arguments."""

    input: str
    sample: int
    seed: int
    output: str
    scaler_output: str
    no_scaler_compare: bool


def parse_args(argv: Sequence[str] | None = None) -> RobustnessArgs:
    p = argparse.ArgumentParser(description="Illumination robustness test.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--sample", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--scaler-output", default=DEFAULT_SCALER_OUTPUT)
    p.add_argument("--no-scaler-compare", action="store_true")
    return cast(RobustnessArgs, p.parse_args(argv))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    summary, scaler_df = run_robustness(
        args.input, args.sample, args.seed, scaler_compare=not args.no_scaler_compare
    )
    if summary is None:
        sys.exit(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {args.output}")
    if scaler_df is not None and not scaler_df.empty:
        scaler_df.to_csv(args.scaler_output, index=False)
        print(scaler_df.to_string(index=False))
        print(f"Saved: {args.scaler_output}")


if __name__ == "__main__":
    main(sys.argv[1:])
