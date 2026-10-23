"""Tests for Min-Max and Z-score trait normalization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from yheart.constants import FEATURES
from yheart.scoring.normalize_minmax import normalize_color_features


def _table(path: Path) -> str:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "filename": [f"img_{i}.png" for i in range(6)],
        **{feature: rng.normal(10, 3, 6) for feature in FEATURES},
    })
    df.to_excel(path, index=False)
    return str(path)


def test_minmax_scales_each_feature_to_unit_interval(tmp_path: Path):
    source = _table(tmp_path / "raw.xlsx")
    output = tmp_path / "minmax.xlsx"
    normalize_color_features(source, str(output), "minmax")
    result = pd.read_excel(output)
    for feature in FEATURES:
        assert result[feature].min() == pytest.approx(0.0)
        assert result[feature].max() == pytest.approx(1.0)


def test_zscore_centres_and_scales(tmp_path: Path):
    source = _table(tmp_path / "raw.xlsx")
    output = tmp_path / "zscore.xlsx"
    normalize_color_features(source, str(output), "zscore")
    result = pd.read_excel(output)
    for feature in FEATURES:
        assert result[feature].mean() == pytest.approx(0.0, abs=1e-9)
        assert result[feature].std(ddof=0) == pytest.approx(1.0)
    # Identifier columns are preserved untouched.
    assert list(result["filename"])[:2] == ["img_0.png", "img_1.png"]


def test_unknown_method_and_missing_columns(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalize_color_features("unused.xlsx", "unused-out.xlsx", "log")
    source = tmp_path / "partial.xlsx"
    pd.DataFrame({"filename": ["a"], "b_mean": [0.5]}).to_excel(source, index=False)
    output = tmp_path / "out.xlsx"
    normalize_color_features(str(source), str(output), "minmax")
    assert not output.exists()
