"""Type-check an expression tree.

Enforces both:
- L2 type signatures (in_types/out_type) — see operator declarations
- hard business constraints (DESIGN.md §4.3):
    - tree depth ≤ max_depth (default 5)
    - leaf count ≤ max_leaves (default 8)
    - financial fields (time_grain == 季/年) require window literal ≥ 1
    - blacklist: AMOUNT / COUNT (dimensionless) is rejected as Ratio args
    - null_semantics warning: fields needing "未授权 vs 无记录" disambiguation
      should appear inside an explicit IfThenElse guard (Phase 0: warn only)

Returns the inferred LogicalType of the root, or raises TypeCheckError with
a precise location (the offending sub-expression's S-expr).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from auto_derivation.config import settings
from auto_derivation.l1_data.registry import MetricRegistry, default_registry
from auto_derivation.l1_data.types import Dtype, LogicalType, NullSemantics, TimeGrain
from auto_derivation.l2_operators.registry import (
    OperatorRegistry,
    default_operator_registry,
)

from .tree import ExprNode, FieldNode, LiteralNode, OpNode


class TypeCheckError(ValueError):
    pass


@dataclass
class TypeCheckResult:
    out_type: LogicalType
    depth: int
    leaves: int
    warnings: list[str] = field(default_factory=list)


def typecheck(
    expr: ExprNode,
    *,
    metric_registry: MetricRegistry | None = None,
    op_registry: OperatorRegistry | None = None,
    max_depth: int | None = None,
    max_leaves: int | None = None,
) -> TypeCheckResult:
    metrics = metric_registry or default_registry()
    ops = op_registry or default_operator_registry()
    max_d = max_depth or settings.max_tree_depth
    max_l = max_leaves or settings.max_tree_leaves

    warnings: list[str] = []
    out_type, depth, leaves = _check(expr, metrics, ops, warnings)

    if depth > max_d:
        raise TypeCheckError(
            f"Tree depth {depth} exceeds limit {max_d} in: {expr.to_sexpr()}"
        )
    if leaves > max_l:
        raise TypeCheckError(
            f"Leaf count {leaves} exceeds limit {max_l} in: {expr.to_sexpr()}"
        )
    return TypeCheckResult(out_type=out_type, depth=depth, leaves=leaves, warnings=warnings)


def _check(
    node: ExprNode,
    metrics: MetricRegistry,
    ops: OperatorRegistry,
    warnings: list[str],
) -> tuple[LogicalType, int, int]:
    if isinstance(node, FieldNode):
        meta = metrics.resolve(node.name)
        if meta.null_semantics == NullSemantics.DISTINGUISH:
            warnings.append(
                f"Field {node.name!r} has ambiguous null semantics; "
                "consider wrapping with IfThenElse(IsAuthorized, ...)."
            )
        return meta.logical_type, 1, 1

    if isinstance(node, LiteralNode):
        return _literal_type(node.value), 1, 1

    if isinstance(node, OpNode):
        return _check_op(node, metrics, ops, warnings)

    raise TypeCheckError(f"Unknown node: {node!r}")  # pragma: no cover


def _literal_type(v: int | float | str) -> LogicalType:
    if isinstance(v, bool):  # bool is a subclass of int, check first
        return LogicalType.BOOL
    if isinstance(v, int):
        return LogicalType.INT
    if isinstance(v, float):
        return LogicalType.NUMERIC
    return LogicalType.CATEGORY


def _check_op(
    node: OpNode,
    metrics: MetricRegistry,
    ops: OperatorRegistry,
    warnings: list[str],
) -> tuple[LogicalType, int, int]:
    op = ops.get(node.op_name)
    sig = op.signature

    expected_field_args = len(sig.in_types)
    expected_lit_args = sig.n_literal_args
    expected_total = expected_field_args + expected_lit_args

    if len(node.children) != expected_total:
        raise TypeCheckError(
            f"Operator {op.name} expects {expected_total} args "
            f"({expected_field_args} typed + {expected_lit_args} literal), "
            f"got {len(node.children)} in: {node.to_sexpr()}"
        )

    # Check the typed (sub-expression) arguments.
    field_args = node.children[:expected_field_args]
    lit_args = node.children[expected_field_args:]

    max_child_depth = 0
    total_leaves = 0
    for i, (arg, expected) in enumerate(zip(field_args, sig.in_types, strict=True)):
        actual, d, lv = _check(arg, metrics, ops, warnings)
        if not _types_compatible(actual, expected):
            raise TypeCheckError(
                f"Operator {op.name} arg {i}: expected {expected}, got {actual} "
                f"in sub-expression: {arg.to_sexpr()}"
            )
        max_child_depth = max(max_child_depth, d)
        total_leaves += lv

    # Literal args must be raw LiteralNodes.
    for j, lit in enumerate(lit_args):
        if not isinstance(lit, LiteralNode):
            raise TypeCheckError(
                f"Operator {op.name} literal arg {j} must be a literal, "
                f"got {lit.to_sexpr()}"
            )
        total_leaves += 1

    # Business constraints.
    _check_business_rules(op.name, field_args, lit_args, metrics, warnings)

    return sig.out_type, max_child_depth + 1, total_leaves


def _types_compatible(actual: LogicalType, expected: LogicalType) -> bool:
    if actual == expected:
        return True
    # Numeric subtypes: INT and RATIO are acceptable wherever NUMERIC is expected.
    return expected == LogicalType.NUMERIC and actual in (LogicalType.INT, LogicalType.RATIO)


def _check_business_rules(
    op_name: str,
    field_args: tuple[ExprNode, ...],
    lit_args: tuple[ExprNode, ...],
    metrics: MetricRegistry,
    warnings: list[str],
) -> None:
    # Window-length sanity: rolling_n operators need positive integer literal.
    if op_name in {"Delta", "PctChange", "Mean", "Std", "ZScore", "Slope", "TsRank", "Count"}:
        if not lit_args:
            return
        n_lit = lit_args[-1]
        if isinstance(n_lit, LiteralNode) and isinstance(n_lit.value, int):
            n = n_lit.value
            if n <= 0:
                raise TypeCheckError(f"{op_name} window must be > 0, got {n}")
            # Financial fields (time_grain quarter/year) need window aligned to quarters.
            for fa in field_args:
                if isinstance(fa, FieldNode):
                    meta = metrics.resolve(fa.name)
                    if meta.time_grain in (TimeGrain.QUARTER, TimeGrain.YEAR) and n < 1:
                        raise TypeCheckError(
                            f"{op_name} on financial field {fa.name!r} "
                            f"requires window ≥ 1 (quarters), got {n}"
                        )

    # Dimensional blacklist: any binary op combining AMOUNT with COUNT directly
    # produces dimensionally meaningless output (金额/计数 has no business unit).
    # We only check direct-child fields (deep symbolic dimensional analysis would
    # over-reject; let GP's IV/redundancy filter the rest).
    if op_name in _DIM_BLACKLIST_BINARY_OPS and len(field_args) == 2:
        a, b = field_args
        ta = _direct_field_dtype(a, metrics)
        tb = _direct_field_dtype(b, metrics)
        if {ta, tb} == {Dtype.AMOUNT, Dtype.COUNT}:
            raise TypeCheckError(
                f"Blacklisted {op_name}: 金额 与 计数 dimensional mismatch in: "
                f"({op_name} {a.to_sexpr()} {b.to_sexpr()})"
            )

    del warnings  # currently unused inside the helper but kept for symmetry


_DIM_BLACKLIST_BINARY_OPS: frozenset[str] = frozenset({
    "Ratio", "Add", "Subtract", "Multiply", "Min", "Max", "Gt2", "Lt2",
})


def _direct_field_dtype(node: ExprNode, metrics: MetricRegistry) -> Dtype | None:
    if isinstance(node, FieldNode):
        return metrics.resolve(node.name).dtype
    return None
