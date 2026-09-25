"""Variable-vs-variable comparison tests: Gt2 / Lt2."""
from __future__ import annotations

import polars as pl
import pytest

from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.expression.typecheck import typecheck
from auto_derivation.l2_operators.base import EvalContext
from auto_derivation.l2_operators.logic import all_logic
from auto_derivation.l2_operators.registry import default_operator_registry


@pytest.fixture
def ops():
    return {op.name: op for op in all_logic()}


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "revenue": [100.0, 80.0, 50.0],
            "receivables": [30.0, 90.0, 200.0],
        }
    )


def test_gt2_basic(df, ops):
    res = df.with_columns(
        ops["Gt2"]
        .compile([pl.col("revenue"), pl.col("receivables")], [], EvalContext())
        .alias("r")
    )
    # revenue > receivables: 100>30=T, 80>90=F, 50>200=F
    assert res["r"].to_list() == [True, False, False]


def test_lt2_is_inverse_of_gt2(df, ops):
    a = df.with_columns(
        ops["Gt2"]
        .compile([pl.col("revenue"), pl.col("receivables")], [], EvalContext())
        .alias("r")
    )["r"].to_list()
    b = df.with_columns(
        ops["Lt2"]
        .compile([pl.col("receivables"), pl.col("revenue")], [], EvalContext())
        .alias("r")
    )["r"].to_list()
    assert a == b


def test_gt2_via_sexpr_parse_and_typecheck():
    """Round-trip: S-expression with Gt2 parses, typechecks, has BOOL out_type."""
    op_names = set(default_operator_registry().names())
    tree = parse_sexpr("(Gt2 净利润 营业收入)", op_names)
    tc = typecheck(tree)
    from auto_derivation.l1_data.types import LogicalType
    assert tc.out_type == LogicalType.BOOL


def test_compose_with_logic(df, ops):
    """Sanity: AND(Gt2(a,b), Lt2(c,d)) — confirms Gt2/Lt2 compose with AND/OR."""
    op_names = set(default_operator_registry().names())
    tree = parse_sexpr("(AND (Gt2 净利润 营业收入) (Lt2 净利润 100))", op_names)
    # Just ensure typecheck doesn't choke; no eval needed.
    typecheck(tree)
