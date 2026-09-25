"""S-expression parser & JSON round-trip tests."""
import pytest

from auto_derivation.expression.tree import (
    FieldNode,
    LiteralNode,
    OpNode,
    from_json,
    parse_sexpr,
    to_canonical_json,
    to_json,
)
from auto_derivation.l2_operators.registry import default_operator_registry

OP_NAMES = set(default_operator_registry().names())


def test_parse_simple():
    tree = parse_sexpr("(GT 营业收入 0)", op_names=OP_NAMES)
    assert isinstance(tree, OpNode)
    assert tree.op_name == "GT"
    assert tree.children == (FieldNode("营业收入"), LiteralNode(0))


def test_parse_nested():
    tree = parse_sexpr("(GT (PctChange 营业收入 1) -0.3)", op_names=OP_NAMES)
    assert isinstance(tree, OpNode)
    inner = tree.children[0]
    assert isinstance(inner, OpNode) and inner.op_name == "PctChange"
    assert tree.children[1] == LiteralNode(-0.3)


def test_sexpr_roundtrip():
    src = "(AND (GT (PctChange 营业收入 1) -0.3) (LT (Slope 净利润 4) 0))"
    tree = parse_sexpr(src, op_names=OP_NAMES)
    rendered = tree.to_sexpr()
    re_parsed = parse_sexpr(rendered, op_names=OP_NAMES)
    assert re_parsed == tree


def test_json_roundtrip():
    tree = parse_sexpr("(GT (PctChange 营业收入 1) -0.3)", op_names=OP_NAMES)
    s = to_json(tree)
    assert from_json(s) == tree


def test_canonical_json_normalises_numeric_literals():
    a = OpNode("GT", (FieldNode("营业收入"), LiteralNode(0)))
    b = OpNode("GT", (FieldNode("营业收入"), LiteralNode(0.0)))
    assert to_json(a) != to_json(b)  # raw form keeps the Python type
    assert to_canonical_json(a) == to_canonical_json(b)
    # Non-integral floats and strings are untouched.
    c = OpNode("GT", (FieldNode("营业收入"), LiteralNode(0.5)))
    assert to_canonical_json(c) != to_canonical_json(a)
    assert '"0.5"' not in to_canonical_json(c)


# --- parser errors carry character offsets ---


def test_error_unknown_operator_has_offset():
    with pytest.raises(ValueError, match=r"Unknown operator.*'Bogus' at char 1"):
        parse_sexpr("(Bogus 营业收入)", op_names=OP_NAMES)


def test_error_missing_close_paren_points_at_opener():
    with pytest.raises(ValueError, match=r"Missing '\)' for '\(' at char 0"):
        parse_sexpr("(GT (PctChange 营业收入 1) -0.3", op_names=OP_NAMES)


def test_error_unexpected_close_paren_has_offset():
    with pytest.raises(ValueError, match=r"Unexpected '\)' at char 0"):
        parse_sexpr(") oops", op_names=OP_NAMES)


def test_error_trailing_token_has_offset():
    with pytest.raises(ValueError, match=r"Unexpected trailing token.*at char 9"):
        parse_sexpr("(GT 营 0) 尾巴", op_names=OP_NAMES)


def test_error_unterminated_string_has_offset():
    with pytest.raises(ValueError, match=r"Unterminated string.*at char 4"):
        parse_sexpr('(GT "营业收入 0)', op_names=OP_NAMES)


def test_error_lone_open_paren():
    with pytest.raises(ValueError, match=r"Missing operator after '\(' at char 0"):
        parse_sexpr("(", op_names=OP_NAMES)
