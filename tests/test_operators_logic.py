import polars as pl
import pytest

from auto_derivation.l2_operators.base import EvalContext
from auto_derivation.l2_operators.logic import _and, _gt, _ifthenelse, _lt, _not, _or


def _eval(df: pl.DataFrame, expr: pl.Expr) -> list:
    return df.with_columns(expr.alias("out"))["out"].to_list()


def test_gt_simple():
    df = pl.DataFrame({"x": [-1.0, 0.0, 1.0]})
    out = _eval(df, _gt([pl.col("x")], [0.0], EvalContext()))
    assert out == [False, False, True]


def test_lt_simple():
    df = pl.DataFrame({"x": [-1.0, 0.0, 1.0]})
    out = _eval(df, _lt([pl.col("x")], [0.0], EvalContext()))
    assert out == [True, False, False]


def test_and_or_not():
    df = pl.DataFrame({"a": [True, True, False], "b": [True, False, False]})
    assert _eval(df, _and([pl.col("a"), pl.col("b")], [], EvalContext())) == [True, False, False]
    assert _eval(df, _or([pl.col("a"), pl.col("b")], [], EvalContext())) == [True, True, False]
    assert _eval(df, _not([pl.col("a")], [], EvalContext())) == [False, False, True]


def test_ifthenelse():
    df = pl.DataFrame({"c": [True, False, True], "a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = _eval(df, _ifthenelse([pl.col("c"), pl.col("a"), pl.col("b")], [], EvalContext()))
    assert out == [pytest.approx(1.0), pytest.approx(20.0), pytest.approx(3.0)]
