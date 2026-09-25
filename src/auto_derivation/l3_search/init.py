"""Type-aware ramped half-and-half initialization.

For each tree:
- pick a target depth in [min_depth, max_depth]
- alternate "full" (all internal nodes are operators down to max depth) and
  "grow" (any node may be a terminal at any depth)
- root must produce BOOL — early-warning rules are predicates over the panel
- typecheck the result and retry up to `max_retries` times if invalid
  (a few BUSINESS_RULE rejections are expected, e.g. window=1 on Slope)
"""
from __future__ import annotations

import numpy as np

from auto_derivation.expression.tree import ExprNode, FieldNode, LiteralNode, OpNode
from auto_derivation.expression.typecheck import TypeCheckError, typecheck
from auto_derivation.l1_data.types import LogicalType

from .primitives import TypedPrimitiveSet

BOOL_OPERATORS = ("GT", "LT", "AND", "OR", "NOT")


class InitError(RuntimeError):
    pass


def _all_inputs_closeable(op_in_types, primset: TypedPrimitiveSet) -> bool:
    """An op is 'closeable' if every input type has at least one terminal
    field — i.e. we can use this op at depth=2 without further recursion."""
    return all(primset.has_field_of_type(t) for t in op_in_types)


def _grow_subtree(
    out_type: LogicalType,
    max_depth: int,
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    *,
    method: str,
) -> ExprNode:
    """Recursively construct a subtree producing `out_type`."""
    fields = primset.fields_of_type(out_type)
    ops = [op for op in primset.operators_producing(out_type) if op.signature.in_types]

    # Terminal slot.
    if max_depth <= 1:
        if fields:
            f = rng.choice(fields)  # type: ignore[arg-type]
            return FieldNode(name=f.name_cn or f.metric_id)
        # No terminal field for this type — close it with a depth-2 op whose
        # inputs are all directly available as fields.
        closeable = [op for op in ops if _all_inputs_closeable(op.signature.in_types, primset)]
        if not closeable:
            raise InitError(f"Cannot close subtree for {out_type} at depth=1")
        return _build_op(rng.choice(closeable), 1, primset, rng, method=method)  # type: ignore[arg-type]

    if not ops:
        if fields:
            f = rng.choice(fields)  # type: ignore[arg-type]
            return FieldNode(name=f.name_cn or f.metric_id)
        raise InitError(f"No operator or field produces {out_type}")

    pick_op = method == "full" or not fields or rng.random() < 0.7
    if pick_op:
        op = rng.choice(ops)  # type: ignore[arg-type]
        return _build_op(op, max_depth, primset, rng, method=method)
    f = rng.choice(fields)  # type: ignore[arg-type]
    return FieldNode(name=f.name_cn or f.metric_id)


def _build_op(
    op,
    max_depth: int,
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    *,
    method: str,
) -> ExprNode:
    children: list[ExprNode] = []
    for in_type in op.signature.in_types:
        child = _grow_subtree(in_type, max_depth - 1, primset, rng, method=method)
        children.append(child)
    for _ in range(op.signature.n_literal_args):
        children.append(LiteralNode(value=primset.sample_literal(op.name, rng)))
    return OpNode(op_name=op.name, children=tuple(children))


def grow_tree(
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    *,
    min_depth: int = 2,
    max_depth: int = 4,
    method: str = "grow",
    root_type: LogicalType = LogicalType.BOOL,
) -> ExprNode:
    """Build one tree. Caller is responsible for typecheck + retry on failure."""
    target_depth = int(rng.integers(min_depth, max_depth + 1))
    return _grow_subtree(root_type, target_depth, primset, rng, method=method)


def ramped_half_and_half(
    primset: TypedPrimitiveSet,
    *,
    pop_size: int,
    min_depth: int = 2,
    max_depth: int = 4,
    rng: np.random.Generator | None = None,
    max_retries: int = 30,
) -> list[ExprNode]:
    """Generate `pop_size` distinct, typecheck-valid trees."""
    rng = rng or np.random.default_rng()
    pop: list[ExprNode] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = pop_size * max_retries
    while len(pop) < pop_size and attempts < max_attempts:
        attempts += 1
        method = "full" if attempts % 2 == 0 else "grow"
        try:
            tree = grow_tree(
                primset, rng, min_depth=min_depth, max_depth=max_depth, method=method
            )
            typecheck(tree)
        except (TypeCheckError, InitError, KeyError):
            continue
        from auto_derivation.expression.tree import to_canonical_json

        k = to_canonical_json(tree)
        if k in seen:
            continue
        seen.add(k)
        pop.append(tree)

    if len(pop) < pop_size:
        raise InitError(
            f"Could only build {len(pop)}/{pop_size} valid trees in {attempts} attempts; "
            "consider widening depth bounds or relaxing constraints."
        )
    return pop
