"""Tests for the genetics table contracts (pandera pilot)."""

import pandas as pd
import pytest
from pandera.errors import SchemaError


from yheart.genetics.schemas import GwasColumns, validate_gwas_table


def _frame(**cols: object) -> pd.DataFrame:
    return pd.DataFrame(dict(cols))


def test_valid_gwas_table_passes() -> None:
    df = _frame(SNP=["s1", "s2"], CHR=["chr1", "chr2"], BP=[100, 200], P=[1e-9, 0.5])
    out = validate_gwas_table(df)
    assert len(out) == 2


def test_missing_column_fails_with_name() -> None:
    df = _frame(SNP=["s1"], CHR=["chr1"], BP=[100])
    with pytest.raises(SchemaError, match="P"):
        validate_gwas_table(df)


def test_wrong_dtype_fails() -> None:
    df = _frame(SNP=["s1"], CHR=["chr1"], BP=["far"], P=[0.1])
    with pytest.raises(SchemaError):
        validate_gwas_table(df)


def test_extra_columns_allowed() -> None:
    df = _frame(
        SNP=["s1"],
        CHR=["chr1"],
        BP=[100],
        P=[0.1],
        BETA=[0.5],
    )
    assert len(validate_gwas_table(df)) == 1


def test_schema_fields_are_declared() -> None:
    assert set(GwasColumns.to_schema().columns) == {"SNP", "CHR", "BP", "P"}
