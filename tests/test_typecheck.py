import pytest

from auto_derivation.expression.tree import FieldNode, LiteralNode, OpNode, parse_sexpr
from auto_derivation.expression.typecheck import TypeCheckError, typecheck
from auto_derivation.l1_data.types import LogicalType
from auto_derivation.l2_operators.registry import default_operator_registry

OP_NAMES = set(default_operator_registry().names())


def test_simple_gt_typechecks_to_bool():
    tree = parse_sexpr("(GT 营业收入 0)", op_names=OP_NAMES)
    res = typecheck(tree)
    assert res.out_type == LogicalType.BOOL
    assert res.depth == 2


def test_pct_change_then_gt():
    tree = parse_sexpr("(GT (PctChange 营业收入 1) -0.3)", op_names=OP_NAMES)
    res = typecheck(tree)
    assert res.out_type == LogicalType.BOOL


def test_arity_mismatch_rejected():
    # GT expects 1 typed arg + 1 literal; here we pass 0 literals.
    bad = OpNode("GT", (FieldNode("营业收入"),))
    with pytest.raises(TypeCheckError, match="expects"):
        typecheck(bad)


def test_unknown_field_rejected():
    bad = OpNode("GT", (FieldNode("不存在的字段XYZ"), LiteralNode(0)))
    with pytest.raises(KeyError):
        typecheck(bad)


def test_zero_window_rejected():
    bad = parse_sexpr("(PctChange 营业收入 0)", op_names=OP_NAMES)
    with pytest.raises(TypeCheckError, match="window"):
        typecheck(bad)


def test_amount_over_count_blacklisted():
    # 营业收入 (金额) / 近期征信查询次数 (计数) — dimensionless nonsense per DESIGN.md §4.3.
    bad = parse_sexpr("(Ratio 营业收入 近期征信查询次数)", op_names=OP_NAMES)
    with pytest.raises(TypeCheckError, match="Blacklisted"):
        typecheck(bad)


@pytest.mark.parametrize(
    "expr",
    [
        "(Add 营业收入 近期征信查询次数)",       # AMOUNT + COUNT
        "(Subtract 近期征信查询次数 营业收入)",  # COUNT - AMOUNT
        "(Multiply 营业收入 近期征信查询次数)",  # AMOUNT × COUNT
        "(Min 营业收入 近期征信查询次数)",       # AMOUNT min COUNT
        "(Max 近期征信查询次数 营业收入)",       # COUNT max AMOUNT
        "(Gt2 营业收入 近期征信查询次数)",       # AMOUNT > COUNT
        "(Lt2 近期征信查询次数 营业收入)",       # COUNT < AMOUNT
    ],
)
def test_dimensional_blacklist_for_new_binary_ops(expr):
    bad = parse_sexpr(expr, op_names=OP_NAMES)
    with pytest.raises(TypeCheckError, match="Blacklisted"):
        typecheck(bad)


def test_arith_with_compatible_dtypes_passes():
    # AMOUNT + AMOUNT, AMOUNT − AMOUNT — both fine.
    typecheck(parse_sexpr("(Add 营业收入 净利润)", op_names=OP_NAMES))
    typecheck(parse_sexpr("(Subtract 营业收入 净利润)", op_names=OP_NAMES))
    typecheck(parse_sexpr("(Gt2 营业收入 净利润)", op_names=OP_NAMES))


def test_depth_limit_enforced():
    # Build a deeply nested PctChange chain that exceeds the default depth=5.
    expr = "(PctChange (PctChange (PctChange (PctChange (PctChange 营业收入 1) 1) 1) 1) 1)"
    tree = parse_sexpr(expr, op_names=OP_NAMES)
    with pytest.raises(TypeCheckError, match="depth"):
        typecheck(tree, max_depth=4)


def test_warning_on_distinguish_field():
    # LOAN_0001 (客户所处地区) has DISTINGUISH null semantics; check the
    # warning by calling `_check` on the bare field.
    from auto_derivation.expression.typecheck import _check

    warnings: list[str] = []
    from auto_derivation.l1_data.registry import default_registry
    from auto_derivation.l2_operators.registry import (
        default_operator_registry as ops_reg,
    )

    _check(FieldNode("客户所处地区"), default_registry(), ops_reg(), warnings)
    assert any("ambiguous null semantics" in w for w in warnings)
