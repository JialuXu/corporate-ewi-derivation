"""Type-safe GP variation: crossover + mutation.

Every produced tree is re-typechecked. If a variation results in an invalid
tree (window-length < 1, depth > limit, etc.), we retry up to `max_retries`
times; if still invalid we return the *original* parent unchanged.

All variation operators are purely functional — they return new ExprNodes
and never mutate the inputs (ExprNode is a frozen dataclass anyway).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auto_derivation.expression.tree import ExprNode, FieldNode, LiteralNode, OpNode
from auto_derivation.expression.typecheck import TypeCheckError, typecheck
from auto_derivation.l1_data.types import LogicalType
from auto_derivation.l2_operators.registry import OperatorRegistry

from .init import grow_tree
from .primitives import TypedPrimitiveSet


@dataclass(frozen=True)
class _NodeRef:
    """A node + its inferred output type + path from the root."""

    node: ExprNode
    out_type: LogicalType
    path: tuple[int, ...]


def _walk(
    node: ExprNode,
    op_registry: OperatorRegistry,
    primset: TypedPrimitiveSet,
    path: tuple[int, ...] = (),
) -> list[_NodeRef]:
    """Yield every node in the tree paired with its output type."""
    out: list[_NodeRef] = []
    out.append(_NodeRef(node=node, out_type=_node_out_type(node, op_registry, primset), path=path))
    if isinstance(node, OpNode):
        for i, child in enumerate(node.children):
            # Skip literal slots — they're not swappable subtrees.
            op = op_registry.get(node.op_name)
            if i >= len(op.signature.in_types):
                continue
            out.extend(_walk(child, op_registry, primset, (*path, i)))
    return out


def _node_out_type(
    node: ExprNode,
    op_registry: OperatorRegistry,
    primset: TypedPrimitiveSet,
) -> LogicalType:
    if isinstance(node, OpNode):
        return op_registry.get(node.op_name).signature.out_type
    if isinstance(node, FieldNode):
        return primset.metric_registry.resolve(node.name).logical_type
    if isinstance(node, LiteralNode):
        return _literal_logical_type(node.value)
    raise TypeError(f"Unknown node type: {type(node).__name__}")  # pragma: no cover


def _literal_logical_type(v: object) -> LogicalType:
    if isinstance(v, bool):
        return LogicalType.BOOL
    if isinstance(v, int):
        return LogicalType.INT
    if isinstance(v, float):
        return LogicalType.NUMERIC
    return LogicalType.CATEGORY


def _replace_at(root: ExprNode, path: tuple[int, ...], new_node: ExprNode) -> ExprNode:
    """Return a new tree with `path`'s subtree replaced by `new_node`."""
    if not path:
        return new_node
    if not isinstance(root, OpNode):
        raise TypeError(f"Cannot descend into non-Op node at path {path}")
    head, *tail = path
    new_child = _replace_at(root.children[head], tuple(tail), new_node)
    new_children = list(root.children)
    new_children[head] = new_child
    return OpNode(op_name=root.op_name, children=tuple(new_children))


def _types_compatible(actual: LogicalType, expected: LogicalType) -> bool:
    if actual == expected:
        return True
    return expected == LogicalType.NUMERIC and actual in (LogicalType.INT, LogicalType.RATIO)


# --- variation operators ---


def crossover(
    parent_a: ExprNode,
    parent_b: ExprNode,
    *,
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    max_retries: int = 10,
) -> tuple[ExprNode, ExprNode]:
    """Subtree crossover. Picks a subtree in A and replaces it with a
    type-compatible subtree from B, and vice versa."""
    op_reg = primset.op_registry
    nodes_a = _walk(parent_a, op_reg, primset)
    nodes_b = _walk(parent_b, op_reg, primset)

    for _ in range(max_retries):
        ra = nodes_a[int(rng.integers(0, len(nodes_a)))]
        # Compatible subtrees from B: ones whose output type matches `ra.out_type`.
        compat_b = [n for n in nodes_b if _types_compatible(n.out_type, ra.out_type)]
        if not compat_b:
            continue
        rb = compat_b[int(rng.integers(0, len(compat_b)))]
        try:
            child_a = _replace_at(parent_a, ra.path, rb.node)
            child_b = _replace_at(parent_b, rb.path, ra.node)
            typecheck(child_a)
            typecheck(child_b)
            return child_a, child_b
        except (TypeCheckError, KeyError):
            continue
    return parent_a, parent_b


def subtree_mutation(
    parent: ExprNode,
    *,
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    max_subtree_depth: int = 3,
    max_retries: int = 10,
) -> ExprNode:
    """Replace a randomly chosen subtree with a freshly grown one of the same
    output type."""
    op_reg = primset.op_registry
    try:
        nodes = _walk(parent, op_reg, primset)
    except KeyError:
        return parent  # parent references unknown field — leave unchanged

    for _ in range(max_retries):
        ref = nodes[int(rng.integers(0, len(nodes)))]
        try:
            new_subtree = grow_tree(
                primset,
                rng,
                min_depth=1,
                max_depth=max_subtree_depth,
                method="grow",
                root_type=ref.out_type,
            )
            child = _replace_at(parent, ref.path, new_subtree)
            typecheck(child)
            return child
        except (TypeCheckError, KeyError):
            continue
    return parent


def point_mutation(
    parent: ExprNode,
    *,
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    max_retries: int = 10,
) -> ExprNode:
    """Swap one operator with another that has the *exact* same signature.
    Cheaper than subtree replacement; useful for fine-grained refinement."""
    op_reg = primset.op_registry
    nodes = _walk(parent, op_reg, primset)
    op_refs = [n for n in nodes if isinstance(n.node, OpNode)]
    if not op_refs:
        return parent

    for _ in range(max_retries):
        ref = op_refs[int(rng.integers(0, len(op_refs)))]
        node = ref.node
        assert isinstance(node, OpNode)
        sig = op_reg.get(node.op_name).signature
        candidates = [
            op
            for op in primset.operators_with_signature(
                sig.in_types, sig.out_type, sig.n_literal_args
            )
            if op.name != node.op_name
        ]
        if not candidates:
            continue
        new_op = candidates[int(rng.integers(0, len(candidates)))]
        new_node = OpNode(op_name=new_op.name, children=node.children)
        try:
            child = _replace_at(parent, ref.path, new_node)
            typecheck(child)
            return child
        except (TypeCheckError, KeyError):
            continue
    return parent


def literal_jitter(
    parent: ExprNode,
    *,
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    max_retries: int = 10,
) -> ExprNode:
    """Resample one literal argument."""
    op_reg = primset.op_registry

    # Find every (parent_path, child_idx, op_name) triple where the child is a literal slot.
    lit_slots: list[tuple[tuple[int, ...], int, str]] = []

    def visit(n: ExprNode, path: tuple[int, ...]) -> None:
        if not isinstance(n, OpNode):
            return
        op = op_reg.get(n.op_name)
        n_field = len(op.signature.in_types)
        for i in range(n_field, n_field + op.signature.n_literal_args):
            lit_slots.append((path, i, n.op_name))
        for i, child in enumerate(n.children[:n_field]):
            visit(child, (*path, i))

    visit(parent, ())
    if not lit_slots:
        return parent

    for _ in range(max_retries):
        path, idx, op_name = lit_slots[int(rng.integers(0, len(lit_slots)))]
        new_val = primset.sample_literal(op_name, rng)
        new_lit = LiteralNode(value=new_val)
        new_op = _set_child(parent, path, idx, new_lit)
        try:
            typecheck(new_op)
            return new_op
        except (TypeCheckError, KeyError):
            continue
    return parent


def _set_child(
    root: ExprNode, op_path: tuple[int, ...], child_idx: int, new_child: ExprNode,
) -> ExprNode:
    """Replace the `child_idx`-th child of the OpNode at `op_path`."""
    if not op_path:
        if not isinstance(root, OpNode):
            raise TypeError(f"Expected OpNode at root, got {type(root).__name__}")
        new_children = list(root.children)
        new_children[child_idx] = new_child
        return OpNode(op_name=root.op_name, children=tuple(new_children))
    if not isinstance(root, OpNode):
        raise TypeError(f"Cannot descend into {type(root).__name__}")
    head, *tail = op_path
    rebuilt = _set_child(root.children[head], tuple(tail), child_idx, new_child)
    new_children = list(root.children)
    new_children[head] = rebuilt
    return OpNode(op_name=root.op_name, children=tuple(new_children))
