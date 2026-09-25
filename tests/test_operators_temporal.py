"""Property tests for temporal operators using small Polars frames."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from auto_derivation.l2_operators.base import EvalContext
from auto_derivation.l2_operators.temporal import (
    _delta,
    _pct_change,
    _rolling_mean,
    _slope,
    _zscore,
)


def _series(values: list[float], entities: list[str] | None = None) -> pl.DataFrame:
    n = len(values)
    if entities is None:
        entities = ["A"] * n
    dates = [date(2024, 1, 1) + timedelta(days=30 * i) for i in range(n)]
    return pl.DataFrame(
        {
            "entity_id": entities,
            "observation_date": dates,
            "x": values,
        }
    ).sort(["entity_id", "observation_date"])


def _eval(df: pl.DataFrame, expr: pl.Expr) -> list:
    return df.with_columns(expr.alias("out"))["out"].to_list()


def test_pct_change_constant_series_is_zero():
    df = _series([10.0] * 5)
    out = _eval(df, _pct_change([pl.col("x")], [1], EvalContext()))
    assert out[0] is None
    assert all(v == pytest.approx(0.0) for v in out[1:])


def test_delta_lag_one_simple():
    df = _series([1.0, 2.0, 4.0, 7.0])
    out = _eval(df, _delta([pl.col("x")], [1], EvalContext()))
    assert out == [None, 1.0, 2.0, 3.0]


def test_rolling_mean_window_3():
    df = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = _eval(df, _rolling_mean([pl.col("x")], [3], EvalContext()))
    assert out[0] == pytest.approx(1.0)  # min_samples=1
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(2.0)
    assert out[4] == pytest.approx(4.0)


def test_zscore_constant_is_nan_or_null():
    df = _series([5.0] * 6)
    out = _eval(df, _zscore([pl.col("x")], [3], EvalContext()))
    # std == 0 → division by zero → None or NaN
    assert all(v is None or v != v or v == 0 for v in out)


def test_slope_increasing_positive():
    df = _series([10.0, 20.0, 30.0, 40.0])
    out = _eval(df, _slope([pl.col("x")], [3], EvalContext()))
    # 3-period window ending at index 3 covers obs [20, 30, 40];
    # endpoint slope = (40 - 20) / (3 - 1) = 10.
    assert out[3] == pytest.approx(10.0)


def test_per_entity_window_isolation():
    df = _series([1.0, 100.0, 2.0, 200.0], entities=["A", "B", "A", "B"])
    out = _eval(df.sort(["entity_id", "observation_date"]),
                _delta([pl.col("x")], [1], EvalContext()))
    # After sort: A:[1,2], B:[100,200]; deltas A:[None,1], B:[None,100]
    assert out == [None, 1.0, None, 100.0]
