"""Cross-source operators — the key operator category (DESIGN.md §4.1).

Used to spot disclosure inconsistencies between two sources reporting on the
same underlying quantity (e.g. self-reported revenue vs. cash-flow inbound).
"""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import EvalContext, Operator, TypeSignature, make_operator


def _ratio(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    # Guard against div-by-zero by returning NULL; downstream IV/KS treats NULL as missing.
    return pl.when(b == 0).then(None).otherwise(a / b)


def _inconsistency(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    """Symmetric relative gap: |a − b| / ((|a| + |b|) / 2). Range [0, 2]."""
    a, b = args
    abs_a = a.abs()
    abs_b = b.abs()
    denom = (abs_a + abs_b) / 2
    return pl.when(denom == 0).then(0.0).otherwise((a - b).abs() / denom)


def all_crosssrc() -> list[Operator]:
    return [
        make_operator(
            "Ratio",
            TypeSignature(
                in_types=(LogicalType.NUMERIC, LogicalType.NUMERIC),
                out_type=LogicalType.RATIO,
            ),
            _ratio,
        ),
        make_operator(
            "Inconsistency",
            TypeSignature(
                in_types=(LogicalType.NUMERIC, LogicalType.NUMERIC),
                out_type=LogicalType.RATIO,
            ),
            _inconsistency,
        ),
    ]
