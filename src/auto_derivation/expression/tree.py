"""Expression tree.

Three node kinds:
- OpNode: an operator application with child nodes + literal args
- FieldNode: reference a metric by name_cn or metric_id (resolved against MetricRegistry)
- LiteralNode: constants used by threshold operators (GT/LT) or windows (Mean N=3)

Two interchangeable serialisations:
- S-expression text (human / CLI)
- dict (program / future GP serialization)

S-expression grammar:

    expr     := atom | "(" head expr* ")"
    head     := IDENT
    atom     := NUMBER | STRING | FIELD_REF
    FIELD_REF := bare token that is neither an operator name nor a number
                 → resolved against MetricRegistry by name_cn first, then metric_id

Examples:
    (GT (PctChange 营业收入 1) -0.3)
    (AND (GT (PctChange 营业收入 1) -0.3) (LT (Slope 资金流净流入 3) 0))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldNode:
    name: str  # may be metric_id or name_cn

    def to_sexpr(self) -> str:
        return _quote_if_needed(self.name)

    def to_dict(self) -> dict:
        return {"kind": "field", "name": self.name}


@dataclass(frozen=True)
class LiteralNode:
    value: int | float | str

    def to_sexpr(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)

    def to_dict(self) -> dict:
        return {"kind": "literal", "value": self.value}


@dataclass(frozen=True)
class OpNode:
    op_name: str
    children: tuple[ExprNode, ...] = field(default_factory=tuple)

    def to_sexpr(self) -> str:
        if not self.children:
            return f"({self.op_name})"
        parts = " ".join(c.to_sexpr() for c in self.children)
        return f"({self.op_name} {parts})"

    def to_dict(self) -> dict:
        return {
            "kind": "op",
            "op": self.op_name,
            "children": [c.to_dict() for c in self.children],
        }


ExprNode = FieldNode | LiteralNode | OpNode


def to_json(node: ExprNode) -> str:
    return json.dumps(node.to_dict(), ensure_ascii=False)


def to_canonical_json(node: ExprNode) -> str:
    """Like `to_json`, but numeric literals are normalised (2.0 → 2) so that
    structurally equivalent trees produce the same string. Use this for cache
    keys / dedup; use `to_json` when the literal's Python type must survive a
    round-trip."""
    return json.dumps(_canonical_dict(node), ensure_ascii=False)


def _canonical_dict(node: ExprNode) -> dict:
    if isinstance(node, LiteralNode):
        v = node.value
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return {"kind": "literal", "value": v}
    if isinstance(node, OpNode):
        return {
            "kind": "op",
            "op": node.op_name,
            "children": [_canonical_dict(c) for c in node.children],
        }
    return node.to_dict()


def from_dict(d: dict) -> ExprNode:
    kind = d["kind"]
    if kind == "field":
        return FieldNode(name=d["name"])
    if kind == "literal":
        return LiteralNode(value=d["value"])
    if kind == "op":
        return OpNode(
            op_name=d["op"],
            children=tuple(from_dict(c) for c in d.get("children", [])),
        )
    raise ValueError(f"Unknown node kind: {kind!r}")


def from_json(s: str) -> ExprNode:
    return from_dict(json.loads(s))


# --- S-expression parser ---


def parse_sexpr(text: str, op_names: set[str]) -> ExprNode:
    """Parse an S-expression. `op_names` is required so that bare tokens can
    be classified as operator (head position) or field (atom position).

    Errors raise `ValueError` with the 0-based character offset into `text`
    so callers (CLI, coach proposal parsing) can point at the culprit."""
    tokens = _tokenise(text)
    node, idx = _parse(tokens, 0, op_names, len(text))
    if idx != len(tokens):
        tok, pos = tokens[idx]
        raise ValueError(f"Unexpected trailing token {tok!r} at char {pos}")
    return node


def _tokenise(text: str) -> list[tuple[str, int]]:
    """Returns (token, start-offset) pairs; offsets are 0-based chars into `text`."""
    out: list[tuple[str, int]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            out.append((c, i))
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 1
            if j >= n:
                raise ValueError(f"Unterminated string literal starting at char {i}")
            out.append((text[i : j + 1], i))
            i = j + 1
            continue
        # bare token
        j = i
        while j < n and not text[j].isspace() and text[j] not in "()":
            j += 1
        out.append((text[i:j], i))
        i = j
    return out


def _parse(
    tokens: list[tuple[str, int]], idx: int, op_names: set[str], text_len: int
) -> tuple[ExprNode, int]:
    if idx >= len(tokens):
        raise ValueError(f"Unexpected end of expression at char {text_len}")
    tok, pos = tokens[idx]
    if tok == "(":
        if idx + 1 >= len(tokens):
            raise ValueError(f"Missing operator after '(' at char {pos}")
        head, head_pos = tokens[idx + 1]
        if head not in op_names:
            raise ValueError(
                f"Unknown operator in head position: {head!r} at char {head_pos}"
            )
        children: list[ExprNode] = []
        i = idx + 2
        while i < len(tokens) and tokens[i][0] != ")":
            child, i = _parse(tokens, i, op_names, text_len)
            children.append(child)
        if i >= len(tokens):
            raise ValueError(f"Missing ')' for '(' at char {pos}")
        return OpNode(op_name=head, children=tuple(children)), i + 1
    if tok == ")":
        raise ValueError(f"Unexpected ')' at char {pos}")
    return _atom_from_token(tok), idx + 1


def _atom_from_token(tok: str) -> ExprNode:
    if tok.startswith('"') and tok.endswith('"'):
        return LiteralNode(value=tok[1:-1])
    # try number
    try:
        if "." in tok or "e" in tok or "E" in tok:
            return LiteralNode(value=float(tok))
        return LiteralNode(value=int(tok))
    except ValueError:
        return FieldNode(name=tok)


def _quote_if_needed(s: str) -> str:
    if any(ch.isspace() or ch in "()\"" for ch in s):
        return f'"{s}"'
    return s
