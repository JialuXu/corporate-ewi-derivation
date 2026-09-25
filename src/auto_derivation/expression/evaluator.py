"""Compile an expression tree to a single `pl.Expr` and evaluate on a panel.

Workflow:
1. typecheck the tree (catches structural / type / business issues early)
2. compile each node to a Polars expression via the operator's `compile`
3. run a single `select` on the wide panel — one pass over the data
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from auto_derivation.l1_data.panel import wide_panel
from auto_derivation.l1_data.registry import MetricRegistry, default_registry
from auto_derivation.l2_operators.base import EvalContext
from auto_derivation.l2_operators.registry import (
    OperatorRegistry,
    default_operator_registry,
)

from .tree import ExprNode, FieldNode, LiteralNode, OpNode
from .typecheck import TypeCheckResult, typecheck


def collect_field_names(node: ExprNode) -> list[str]:
    """Return all unique FieldNode names referenced by the tree."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(n: ExprNode) -> None:
        if isinstance(n, FieldNode):
            if n.name not in seen:
                seen.add(n.name)
                out.append(n.name)
        elif isinstance(n, OpNode):
            for c in n.children:
                visit(c)

    visit(node)
    return out


def compile_expr(
    node: ExprNode,
    *,
    metrics: MetricRegistry,
    ops: OperatorRegistry,
    ctx: EvalContext,
) -> pl.Expr:
    """Compile a tree to a single `pl.Expr`. Field names are resolved against
    `metrics` and rewritten to their `metric_id` (which is the column name in
    the wide panel)."""
    if isinstance(node, FieldNode):
        meta = metrics.resolve(node.name)
        return pl.col(meta.metric_id)

    if isinstance(node, LiteralNode):
        return pl.lit(node.value)

    if isinstance(node, OpNode):
        op = ops.get(node.op_name)
        n_field = len(op.signature.in_types)
        compiled_field_args = [
            compile_expr(c, metrics=metrics, ops=ops, ctx=ctx) for c in node.children[:n_field]
        ]
        literal_values: Sequence = [
            _literal_value(c) for c in node.children[n_field:]
        ]
        return op.compile(compiled_field_args, literal_values, ctx)

    raise TypeError(f"Unknown ExprNode type: {type(node).__name__}")  # pragma: no cover


def _literal_value(node: ExprNode) -> int | float | str:
    if isinstance(node, LiteralNode):
        return node.value
    raise TypeError(f"Expected literal, got: {node}")


def evaluate(
    expr: ExprNode,
    panel_path: Path | None = None,
    *,
    panel: pl.DataFrame | None = None,
    metric_registry: MetricRegistry | None = None,
    op_registry: OperatorRegistry | None = None,
    column_name: str = "expr_value",
) -> tuple[pl.DataFrame, TypeCheckResult]:
    """End-to-end: typecheck, load required columns, compile, evaluate.

    Pass either `panel_path` (the long parquet is pivoted per call) or a
    pre-pivoted wide `panel` — batch callers evaluating many expressions
    should pivot once and pass `panel` to skip the per-call
    Parquet → DuckDB → Arrow → Polars chain.

    Returns (DataFrame[entity_id, observation_date, industry, expr_value],
             TypeCheckResult).
    """
    metrics = metric_registry or default_registry()
    ops = op_registry or default_operator_registry()

    tc = typecheck(expr, metric_registry=metrics, op_registry=ops)

    # Resolve referenced metric_ids and pivot only those columns.
    metric_ids = [metrics.resolve(n).metric_id for n in collect_field_names(expr)]
    if panel is None:
        if panel_path is None:
            raise ValueError("evaluate() needs either panel_path or panel")
        panel = wide_panel(panel_path, metric_ids=metric_ids)
    else:
        missing = [m for m in metric_ids if m not in panel.columns]
        if missing:
            raise ValueError(
                f"Pre-pivoted panel is missing columns required by the "
                f"expression: {missing}"
            )

    ctx = EvalContext()
    compiled = compile_expr(expr, metrics=metrics, ops=ops, ctx=ctx)

    result = panel.with_columns(compiled.alias(column_name)).select(
        ["entity_id", "observation_date", "industry", column_name]
    )
    return result, tc
