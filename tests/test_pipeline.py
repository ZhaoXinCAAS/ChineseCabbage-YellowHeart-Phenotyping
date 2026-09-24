"""Unit tests for the pure-function core of the yellow-heart pipeline.

These tests exercise the non-IO numerical helpers (ExR computation, circular
hue mean, polygon clipping) on synthetic data, plus the Fisher-score routine
on a tiny in-memory table, so they run without any image files or Excel inputs.
"""

import math
from pathlib import Path

# Make the `src` layout importable when running `pytest` without an
# installed editable wheel (with `uv sync` the package is installed, so this
# is only a fallback).

import numpy as np
import pandas as pd
import pytest


from yheart.imaging import crop_heads
from yheart.imaging import (
    extract_color_features as Ycfe,
)
from yheart.imaging import (
    segment_yellow_heart as Seg,
)
from yheart.scoring import fisher_scores


def test_round_and_clip_polygon_clamps_to_bounds():
    pts = [[-5, -3], [100, 0], [0, 200]]
    out = Seg.round_and_clip_polygon(pts, w=50, h=80)
    assert out.shape == (3, 2)
    assert out[:, 0].min() >= 0 and out[:, 0].max() <= 49
    assert out[:, 1].min() >= 0 and out[:, 1].max() <= 79


def test_round_and_clip_polygon_drops_malformed():
    out = Seg.round_and_clip_polygon([[1], [2, 3]], w=10, h=10)
    assert out.shape == (1, 2)


def test_compute_exr_matches_reference_formula():
    # ExR = (1.4*R - G) / (R + G + B), with S=0 guarded.
    img = np.array([[[10, 20, 30], [0, 0, 0]]], dtype=np.uint8)
    exr = Seg.compute_exr_per_pixel(img)
    r, g, b = 30.0, 20.0, 10.0
    expected = (1.4 * r - g) / (r + g + b + Seg.EPS)
    assert math.isclose(float(exr[0, 0]), expected, rel_tol=1e-9)
    # Black pixel (S=0) must be finite, not NaN.
    assert np.isfinite(exr[0, 1])


def test_exr_mean_equivalent_to_pixel_formula():
    R = np.array([30.0, 100.0])
    G = np.array([20.0, 50.0])
    B = np.array([10.0, 25.0])
    got = Ycfe.compute_ExR_mean(R, G, B)
    s = R + G + B
    expected = float(np.mean((1.4 * R - G) / s))
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_circular_mean_degrees_handles_wraparound():
    # Two hues at 358 and 2 degrees (OpenCV stores half-degrees: 179 and 1).
    deg = Ycfe.circular_mean_degrees(np.array([179, 1]))
    assert abs(deg - 0.0) < 1.0 or abs(deg - 360.0) < 1.0


def test_circular_mean_degrees_empty_is_nan():
    assert math.isnan(Ycfe.circular_mean_degrees(np.array([], dtype=np.float64)))


def test_fisher_scores_on_separable_synthetic_data(tmp_path: Path):
    # A single feature with well-separated class means AND tiny within-class
    # noise -> a strictly positive Fisher score, with positive response
    # direction for the "deep" class.
    rng = np.random.default_rng(0)
    n = 20
    b_means = np.concatenate([
        rng.normal(0.05, 0.01, n),
        rng.normal(0.50, 0.01, n),
        rng.normal(0.95, 0.01, n),
    ]).clip(0, 1)
    df = pd.DataFrame({
        "filename": [f"a{i}.jpg" for i in range(n * 3)],
        "b_mean": b_means,
        # Other features: tiny noise but identical distributions across
        # classes -> Fisher score ~0. Small noise avoids MinMaxScaler divide-by-0.
        **{f: (0.3 + rng.normal(0, 0.001, n * 3)) for f in fisher_scores.FEATURES if f != "b_mean"},
    })
    features_file = tmp_path / "features.xlsx"
    df.to_excel(features_file, index=False)

    # Build labeled folders by filename.
    root = tmp_path / "labels"
    for idx, label in enumerate(fisher_scores.LABEL_ORDER):
        d = root / label
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"a{i + idx * n}.jpg").write_text("")

    out = tmp_path / "fisher.xlsx"
    fisher_scores.compute_fisher_scores(str(features_file), str(root), str(out))
    result = pd.read_excel(out)

    assert "Feature_Name" in result.columns
    assert "Fisher_Score" in result.columns
    assert "Response_Direction" in result.columns
    row = result.loc[result["Feature_Name"] == "b_mean"].iloc[0]
    assert row["Response_Direction"] == "Positive (+)"
    assert row["Fisher_Score"] > 0
    # b_mean should be the most discriminative feature here.
    assert row["Fisher_Score"] == result["Fisher_Score"].max()
    assert result.shape[0] == len(fisher_scores.FEATURES)


def test_fisher_scores_no_labeled_samples_is_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    features_file = tmp_path / "features.xlsx"
    df = pd.DataFrame({
        "filename": ["x.jpg"],
        "b_mean": [0.1],
        **{f: [0.2] for f in fisher_scores.FEATURES if f != "b_mean"},
    })
    df.to_excel(features_file, index=False)
    root = tmp_path / "empty_labels"
    root.mkdir()
    out = tmp_path / "fisher.xlsx"
    # Must not raise; must print a diagnostic and not write the output file.
    fisher_scores.compute_fisher_scores(str(features_file), str(root), str(out))
    captured = capsys.readouterr()
    assert "No labeled samples" in captured.out
    assert not out.exists()


def test_crop_parse_args_defaults():
    args = crop_heads.parse_args([])
    assert args.input == crop_heads.DEFAULT_JSON_FOLDER
    assert args.output == crop_heads.DEFAULT_OUTPUT_DIR
    assert args.expand_ratio == crop_heads.DEFAULT_EXPAND_RATIO
