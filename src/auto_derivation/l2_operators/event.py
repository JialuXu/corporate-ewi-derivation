"""Event-counting operators."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import EvalContext, Operator, TypeSignature, make_operator


def _count(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    """Count of `True` (or non-zero) values over the past N observations."""
    (cond,) = args
    (n,) = literals
    return (
        cond.cast(pl.Int8)
        .rolling_sum(window_size=int(n), min_samples=1)
        .over("entity_id")
    )


def _days_since(args: Sequence[pl.Expr], _literals: Sequence, ctx: EvalContext) -> pl.Expr:
    """Days since the most recent True value of `cond` (NULL if never)."""
    (cond,) = args
    date_col = pl.col(ctx.date_col)
    last_true_date = (
        pl.when(cond).then(date_col).otherwise(None)
        .forward_fill()
        .over("entity_id")
    )
    return (date_col - last_true_date).dt.total_days()


def all_event() -> list[Operator]:
    return [
        make_operator(
            "Count",
            TypeSignature(
                in_types=(LogicalType.BOOL,),
                out_type=LogicalType.INT,
                n_literal_args=1,
                window_kind="rolling_n",
            ),
            _count,
        ),
        make_operator(
            "DaysSince",
            TypeSignature(
                in_types=(LogicalType.BOOL,),
                out_type=LogicalType.NUMERIC,
                n_literal_args=0,
                window_kind="none",
            ),
            _days_since,
        ),
    ]
