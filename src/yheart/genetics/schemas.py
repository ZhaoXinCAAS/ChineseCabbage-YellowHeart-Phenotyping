"""Shared table contracts (pandera schemas) for cross-module IO boundaries.

Each schema pins the columns one stage promises the next: validation fails
fast with a named error instead of a cryptic `KeyError` deep in the
pipeline. Schemas use `Field()` assignments (required by type checkers)
and stay permissive to extra columns (`strict=False` default).
"""

from __future__ import annotations

from typing import Literal, overload

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class GwasColumns(pa.DataFrameModel):
    """SNP/CHR/BP/P contract shared by gwas/manhattan_plot/annotate_genes."""

    # Identifiers: PLINK writes them unquoted (CHR often integer-coded),
    # so coerce to str instead of rejecting real tool output.
    SNP: Series[str] = pa.Field(coerce=True)
    CHR: Series[str] = pa.Field(coerce=True)
    BP: Series[int] = pa.Field()
    P: Series[float] = pa.Field()


def validate_gwas_table(df: pd.DataFrame) -> DataFrame[GwasColumns]:
    """Validate a GWAS results table; raise ``SchemaError`` naming the fault."""
    return GwasColumns.validate(df)


@overload
def gwas_col(df: DataFrame[GwasColumns], name: Literal["BP"]) -> Series[int]: ...
@overload
def gwas_col(df: DataFrame[GwasColumns], name: Literal["P"]) -> Series[float]: ...
@overload
def gwas_col(df: DataFrame[GwasColumns], name: Literal["SNP", "CHR"]) -> Series[str]: ...
def gwas_col(df: DataFrame[GwasColumns], name: str):
    """Dtype-precise column access (``df[name]`` alone only yields ``Series[Any]``)."""
    return df[name]
