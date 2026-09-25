"""Logical / threshold operators. Output type is BOOL except IfThenElse."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import EvalContext, Operator, TypeSignature, make_operator


def _gt(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (thr,) = literals
    return x > float(thr)


def _lt(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (thr,) = literals
    return x < float(thr)


def _and(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a & b


def _or(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a | b


def _not(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (a,) = args
    return ~a


def _ifthenelse(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    cond, a, b = args
    return pl.when(cond).then(a).otherwise(b)


def _gt2(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a > b


def _lt2(args: Sequence[pl.Expr], _literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    a, b = args
    return a < b


_NUM_BOOL_THR = TypeSignature(
    in_types=(LogicalType.NUMERIC,),
    out_type=LogicalType.BOOL,
    n_literal_args=1,
    window_kind="none",
)
_NUM_NUM_BOOL = TypeSignature(
    in_types=(LogicalType.NUMERIC, LogicalType.NUMERIC),
    out_type=LogicalType.BOOL,
)


def all_logic() -> list[Operator]:
    return [
        make_operator("GT", _NUM_BOOL_THR, _gt),
        make_operator("LT", _NUM_BOOL_THR, _lt),
        make_operator(
            "AND",
            TypeSignature(
                in_types=(LogicalType.BOOL, LogicalType.BOOL),
                out_type=LogicalType.BOOL,
            ),
            _and,
        ),
        make_operator(
            "OR",
            TypeSignature(
                in_types=(LogicalType.BOOL, LogicalType.BOOL),
                out_type=LogicalType.BOOL,
            ),
            _or,
        ),
        make_operator(
            "NOT",
            TypeSignature(
                in_types=(LogicalType.BOOL,),
                out_type=LogicalType.BOOL,
            ),
            _not,
        ),
        make_operator(
            "IfThenElse",
            TypeSignature(
                in_types=(LogicalType.BOOL, LogicalType.NUMERIC, LogicalType.NUMERIC),
                out_type=LogicalType.NUMERIC,
            ),
            _ifthenelse,
        ),
        # Variable-vs-variable comparison (no literal threshold).
        make_operator("Gt2", _NUM_NUM_BOOL, _gt2),
        make_operator("Lt2", _NUM_NUM_BOOL, _lt2),
    ]
