"""Explanation runner — glue between expression, evaluator, LLM, and consistency.

Public entry point: `explain(expr, ...)`. The runner:
  1. typechecks the tree
  2. evaluates it on the panel to get IV/KS/Gini (so the prompt has real numbers)
  3. assembles the prompt and calls the LLM client
  4. parses the JSON response into ExplanationCard (pydantic validates field constraints)
  5. runs consistency_check
  6. returns ExplanationResult
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from auto_derivation.evaluation.metrics import score
from auto_derivation.expression.evaluator import evaluate
from auto_derivation.expression.tree import ExprNode
from auto_derivation.l1_data.panel import join_labels
from auto_derivation.l1_data.registry import MetricRegistry, default_registry

from .card import ExplanationCard, ExplanationResult
from .consistency import consistency_check
from .llm_client import LLMClient, parse_json_response
from .prompt import SYSTEM_PROMPT, PromptContext, assemble_user_prompt


def explain(
    expr: ExprNode,
    *,
    panel_path: Path | None = None,
    labels_path: Path,
    label: str,
    llm_client: LLMClient,
    panel: pl.DataFrame | None = None,
    metric_registry: MetricRegistry | None = None,
    similar_metric_name: str | None = None,
    similar_correlation: float | None = None,
) -> ExplanationResult:
    """Pass a pre-pivoted wide `panel` when explaining many expressions in a
    batch — it skips the per-call pivot inside `evaluate()`."""
    registry = metric_registry or default_registry()

    # 1+2: evaluate on the panel.
    df, _tc = evaluate(expr, panel_path, panel=panel, metric_registry=registry)
    df = join_labels(df, labels_path, label_name=label)
    df = df.filter(pl.col("expr_value").is_not_null() & pl.col(label).is_not_null())
    if df.schema["expr_value"].is_numeric():
        df = df.filter(pl.col("expr_value").is_finite())
    metrics = score(df, value_col="expr_value", label_col=label)

    # 3: prompt + LLM call.
    ctx = PromptContext(
        expr=expr,
        metrics=metrics,
        registry=registry,
        similar_metric_name=similar_metric_name,
        similar_correlation=similar_correlation,
    )
    user = assemble_user_prompt(ctx)
    raw = llm_client.complete(system=SYSTEM_PROMPT, user=user)

    # 4: parse + validate.
    parsed = parse_json_response(raw)
    card = ExplanationCard.model_validate(parsed)

    # 5: consistency check.
    warnings = consistency_check(card, expr, metrics, registry)

    return ExplanationResult(card=card, consistency_warnings=warnings, raw_llm_output=raw)
