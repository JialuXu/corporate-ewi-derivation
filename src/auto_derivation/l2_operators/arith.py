"""Binary arithmetic operators.

Five element-wise operators on numeric pairs: Add / Subtract / Multiply / Min / Max.
All have signature (NUMERIC, NUMERIC) → NUMERIC with no literal arg.

Min/Max use Polars' horizontal aggregations.

Division is provided by `Ratio(a, b)` in crosssrc.py, which guards against
div-by-zero.
"""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import EvalContext, Operator, TypeSignature, make_operator


def _add(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a + b


def _subtract(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a - b


def _multiply(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a * b


def _min(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return pl.min_horizontal(a, b)


def _max(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return pl.max_horizontal(a, b)


_NUM_BIN = TypeSignature(
    in_types=(LogicalType.NUMERIC, LogicalType.NUMERIC),
    out_type=LogicalType.NUMERIC,
)


def all_arith() -> list[Operator]:
    return [
        make_operator("Add", _NUM_BIN, _add),
        make_operator("Subtract", _NUM_BIN, _subtract),
        make_operator("Multiply", _NUM_BIN, _multiply),
        make_operator("Min", _NUM_BIN, _min),
        make_operator("Max", _NUM_BIN, _max),
    ]
