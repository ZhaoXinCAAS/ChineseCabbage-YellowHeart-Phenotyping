"""Tests for the shared color-feature vocabulary."""




from yheart import constants as C
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
