"""End-to-end evaluator tests on the synthetic panel.

This file is the single end-to-end regression check that the
evaluator pipeline surfaces our injected signal at IV ≥ 0.10 even on the
small 120-customer fixture. (On the full 500-customer panel this comfortably
exceeds 0.20.)
"""
from __future__ import annotations

import polars as pl
import pytest

from auto_derivation.evaluation.metrics import score
from auto_derivation.expression.evaluator import evaluate
from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.l1_data.panel import join_labels, wide_panel
from auto_derivation.l2_operators.registry import default_operator_registry

OP_NAMES = set(default_operator_registry().names())


def _eval_and_score(panel_path, labels_path, expr_text: str, label: str):
    tree = parse_sexpr(expr_text, op_names=OP_NAMES)
    df, _ = evaluate(tree, panel_path)
    df = join_labels(df, labels_path, label)
    df = df.filter(pl.col("expr_value").is_not_null() & pl.col("expr_value").is_finite())
    return score(df, value_col="expr_value", label_col=label)


def test_evaluator_runs_end_to_end(synthetic_paths):
    panel, labels = synthetic_paths
    metrics = _eval_and_score(panel, labels, "(PctChange 营业收入 12)", "Y_overdue_3m")
    assert metrics.n > 0  # something was scored


def test_revenue_decline_detects_bad_customers(synthetic_paths):
    """Injection regression: bad customers have revenue decline starting month 12.
    PctChange(营业收入, 12) should be strongly negative for bad customers, so
    the IV against Y_overdue_3m must be substantial."""
    panel, labels = synthetic_paths
    metrics = _eval_and_score(panel, labels, "(PctChange 营业收入 12)", "Y_overdue_3m")
    assert metrics.iv >= 0.10, f"IV too low: {metrics.iv:.4f}"
    assert metrics.ks >= 0.15, f"KS too low: {metrics.ks:.4f}"


def test_judicial_count_predicts_npl(synthetic_paths):
    """Bad customers have judicial cases injected in months 14-18, leading to
    NPL events later. Count(GT(JUD, 0), 6) over a 6-month window should be IV-positive."""
    panel, labels = synthetic_paths
    expr = "(Count (GT 近一年被执行案件数量 0) 6)"
    metrics = _eval_and_score(panel, labels, expr, "Y_npl_12m")
    assert metrics.iv >= 0.05, f"IV too low: {metrics.iv:.4f}"


def test_evaluate_with_prepivoted_panel_matches_path(synthetic_paths):
    """Batch callers pivot once and pass `panel=`; result must be identical
    to the per-call pivot path."""
    panel_path, _ = synthetic_paths
    tree = parse_sexpr("(PctChange 营业收入 12)", op_names=OP_NAMES)
    panel = wide_panel(panel_path)

    from_path, _ = evaluate(tree, panel_path)
    from_panel, _ = evaluate(tree, panel=panel)

    key = ["entity_id", "observation_date"]
    assert from_path.sort(key).equals(from_panel.sort(key))


def test_evaluate_requires_path_or_panel():
    tree = parse_sexpr("(PctChange 营业收入 12)", op_names=OP_NAMES)
    with pytest.raises(ValueError, match="panel_path or panel"):
        evaluate(tree)


def test_evaluate_prepivoted_panel_missing_column(synthetic_paths):
    panel_path, _ = synthetic_paths
    tree = parse_sexpr("(PctChange 营业收入 12)", op_names=OP_NAMES)
    # Drop 营业收入 (FIN_0001) so the panel can't satisfy the expression.
    panel = wide_panel(panel_path).drop("FIN_0001")
    with pytest.raises(ValueError, match="FIN_0001"):
        evaluate(tree, panel=panel)


def test_random_capital_has_no_signal(synthetic_paths):
    """Negative control: 注册资本 is independent of bad/good — IV should be ~0."""
    panel, labels = synthetic_paths
    metrics = _eval_and_score(panel, labels, "注册资本", "Y_overdue_3m")
    assert metrics.iv < 0.05, f"Unexpected signal in noise: IV={metrics.iv:.4f}"
