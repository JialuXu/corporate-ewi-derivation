"""Temporal-change and temporal-anomaly operators.

Window semantics: operations are computed *over each entity's time series*,
i.e. `.over("entity_id")`. Window length is given as the number of rows
(observations) — for a monthly panel, N=3 means 3 months.
"""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import EvalContext, Operator, TypeSignature, make_operator


def _entity_window(expr: pl.Expr) -> pl.Expr:
    """Order-by-date window over each entity. Polars `.over` requires the
    inputs to already be sorted by date within each entity group, which
    `wide_panel()` guarantees."""
    return expr.over("entity_id")


def _delta(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (n,) = literals
    return _entity_window(x - x.shift(int(n)))


def _pct_change(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (n,) = literals
    prev = x.shift(int(n))
    return _entity_window((x - prev) / prev)


def _rolling_mean(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (n,) = literals
    return _entity_window(x.rolling_mean(window_size=int(n), min_samples=1))


def _rolling_std(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (n,) = literals
    return _entity_window(x.rolling_std(window_size=int(n), min_samples=2))


def _zscore(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    (x,) = args
    (n,) = literals
    n_int = int(n)
    mean = x.rolling_mean(window_size=n_int, min_samples=1)
    std = x.rolling_std(window_size=n_int, min_samples=2)
    return _entity_window((x - mean) / std)


def _slope(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    """N-period linear trend slope. Approximated by simple endpoint slope:
    (x_t − x_{t-N+1}) / (N − 1). Cheap and adequate for Phase 0 fitness; a
    full OLS variant can swap in later without touching callers."""
    (x,) = args
    (n,) = literals
    n_int = int(n)
    if n_int < 2:
        raise ValueError("Slope requires window >= 2")
    return _entity_window((x - x.shift(n_int - 1)) / (n_int - 1))


def _ts_rank(args: Sequence[pl.Expr], literals: Sequence, _ctx: EvalContext) -> pl.Expr:
    """Rank of the current value within the past N observations, in [0, 1]."""
    (x,) = args
    (n,) = literals
    n_int = int(n)
    rolled = x.rolling_max(window_size=n_int, min_samples=1)  # placeholder for shape
    # Polars supports rolling_quantile; rank via element-wise comparison fold.
    # Simpler: compute (count of past values <= current) / N using rolling sum
    # of indicator. We do this with a window-aware approach below.
    # For Phase 0 we use rolling_quantile inverse: TsRank ≈ rolling rank of last value.
    del rolled
    # Use rolling map: percentile of the last value within the window.
    return _entity_window(
        x.rolling_map(
            lambda s: ((s <= s[-1]).sum() / s.len()),
            window_size=n_int,
            min_samples=1,
        )
    )


# --- signatures ---

_WINDOW_NUM = TypeSignature(
    in_types=(LogicalType.NUMERIC,),
    out_type=LogicalType.NUMERIC,
    n_literal_args=1,
    window_kind="rolling_n",
)

_WINDOW_RATIO = TypeSignature(
    in_types=(LogicalType.NUMERIC,),
    out_type=LogicalType.RATIO,
    n_literal_args=1,
    window_kind="rolling_n",
)


def all_temporal() -> list[Operator]:
    return [
        make_operator("Delta", _WINDOW_NUM, _delta),
        make_operator("PctChange", _WINDOW_RATIO, _pct_change),
        make_operator("Mean", _WINDOW_NUM, _rolling_mean),
        make_operator("Std", _WINDOW_NUM, _rolling_std),
        make_operator("ZScore", _WINDOW_NUM, _zscore),
        make_operator("Slope", _WINDOW_NUM, _slope),
        make_operator("TsRank", _WINDOW_RATIO, _ts_rank),
    ]
