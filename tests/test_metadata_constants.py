"""Tests for constants vocabulary + validate_metadata (R2-1/Q3)."""

from typing import cast

import pandas as pd
import pytest


from yheart import constants as C
from yheart.imaging.validate_metadata import audit
from yheart.scoring.fisher_scores import FEATURES as F_FISHER
from yheart.scoring.normalize_minmax import (
    COLOR_FEATURES as F_MINMAX,
)


def test_feature_vocabulary_is_consistent():
    assert set(F_FISHER) == set(F_MINMAX) == set(C.FEATURES)
    assert C.abbr("Yellow_Ratio") == "YR"
    assert C.abbr("Yellow_score") == "YS"
    assert C.abbr("CYS") == "CYS"
    for f in C.FEATURES:
        assert f in C.SYMBOLS


def test_metadata_audit_flags_long_interval():
    df = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3"],
        "genotype": ["G1", "G1", "G2"],
        "harvest_time": ["2024-10-01 08:00"] * 3,
        "imaging_time": ["2024-10-01 10:00", "2024-10-05 10:00", "2024-10-01 09:00"],
    })
    table, summary = audit(df, max_interval_h=48.0)
    assert summary["n"] == 3 and summary["genotypes"] == 2
    assert summary["median_interval_h"] == 2.0
    assert summary["n_flagged"] == 1
    assert "long-interval" in cast(str, table.loc[1, "flag"])


def test_metadata_missing_columns_is_clean():
    with pytest.raises(ValueError, match="Missing required columns"):
        audit(pd.DataFrame({"sample_id": ["s1"]}))
