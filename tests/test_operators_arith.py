"""Arithmetic operator tests: Add / Subtract / Multiply / Min / Max."""
from __future__ import annotations

import polars as pl
import pytest

from auto_derivation.l2_operators.arith import all_arith
from auto_derivation.l2_operators.base import EvalContext


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "entity_id": ["A"] * 4,
            "observation_date": pl.date_range(
                start=pl.date(2024, 1, 1),
                end=pl.date(2024, 4, 1),
                interval="1mo",
                eager=True,
            ),
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [3.0, 5.0, 6.0, 8.0],
        }
    )


@pytest.fixture
def ops():
    return {op.name: op for op in all_arith()}


def test_add(df, ops):
    res = df.with_columns(
        ops["Add"].compile([pl.col("x"), pl.col("y")], [], EvalContext()).alias("r")
    )
    assert res["r"].to_list() == [13.0, 25.0, 36.0, 48.0]


def test_subtract(df, ops):
    res = df.with_columns(
        ops["Subtract"].compile([pl.col("x"), pl.col("y")], [], EvalContext()).alias("r")
    )
    assert res["r"].to_list() == [7.0, 15.0, 24.0, 32.0]


def test_multiply(df, ops):
    res = df.with_columns(
        ops["Multiply"].compile([pl.col("x"), pl.col("y")], [], EvalContext()).alias("r")
    )
    assert res["r"].to_list() == [30.0, 100.0, 180.0, 320.0]


def test_min(df, ops):
    res = df.with_columns(
        ops["Min"].compile([pl.col("x"), pl.col("y")], [], EvalContext()).alias("r")
    )
    assert res["r"].to_list() == [3.0, 5.0, 6.0, 8.0]


def test_max(df, ops):
    res = df.with_columns(
        ops["Max"].compile([pl.col("x"), pl.col("y")], [], EvalContext()).alias("r")
    )
    assert res["r"].to_list() == [10.0, 20.0, 30.0, 40.0]


def test_subtract_propagates_null(ops):
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, None]})
    res = df.with_columns(
        ops["Subtract"].compile([pl.col("a"), pl.col("b")], [], EvalContext()).alias("r")
    )
    # Polars convention: null in either side → null result.
    assert res["r"].to_list() == [-9.0, None, None]


def test_compose_subtract_inside_lt(ops):
    """Sanity: 净资产 = 总资产 − 总负债 → 检查是否 < 0 (resvent)."""
    df = pl.DataFrame({"assets": [100.0, 50.0, 200.0], "debt": [60.0, 70.0, 80.0]})
    nav = ops["Subtract"].compile([pl.col("assets"), pl.col("debt")], [], EvalContext())
    res = df.with_columns(insolvent=(nav < 0))
    assert res["insolvent"].to_list() == [False, True, False]
