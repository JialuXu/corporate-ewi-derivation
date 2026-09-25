"""Consistency check tests."""
from __future__ import annotations

import pytest

from auto_derivation.evaluation.metrics import DiscriminationMetrics
from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.l1_data.registry import default_registry
from auto_derivation.l2_operators.registry import default_operator_registry
from auto_derivation.l4_explain.card import ExplanationCard
from auto_derivation.l4_explain.consistency import consistency_check


@pytest.fixture(scope="module")
def reg():
    return default_registry()


@pytest.fixture(scope="module")
def op_names():
    return set(default_operator_registry().names())


def _card(**fields):
    base = {
        "metric_name_cn": "测试指标",
        "business_explanation": "占位说明",
        "use_cases": "对公客户",
        "diff_vs_existing": "无",
        "threshold_advice": "无",
        "review_checklist": ["复核1"],
    }
    base.update(fields)
    return ExplanationCard.model_validate(base)


def _metrics(iv=0.3, ks=0.2):
    return DiscriminationMetrics(
        iv=iv, ks=ks, gini=0.4, n=1000, n_pos=50, trigger_rate=0.05
    )


def test_no_issues_when_explanation_matches_tree(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    card = _card(
        business_explanation="当 营业收入 高于阈值时触发，提示经营异常。",
        threshold_advice="阈值 > 0",
    )
    issues = consistency_check(card, expr, _metrics(), reg)
    assert issues == []


def test_field_hallucination_detected(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    # Mention 净利润 (a real metric) but it's not in the tree.
    card = _card(
        business_explanation="当 净利润 显著上升时触发预警。",
    )
    issues = consistency_check(card, expr, _metrics(), reg)
    assert any("净利润" in i for i in issues), f"expected hallucination flag, got {issues}"


def test_referenced_fields_valid_passes(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    card = _card(
        business_explanation="当 营业收入 高于阈值时触发。",
        referenced_fields=["营业收入"],
    )
    issues = consistency_check(card, expr, _metrics(), reg)
    assert issues == []


def test_referenced_fields_hallucination_detected(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    card = _card(referenced_fields=["净利润"])
    issues = consistency_check(card, expr, _metrics(), reg)
    assert any("净利润" in i and "并未引用" in i for i in issues), issues


def test_referenced_fields_unknown_metric_detected(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    card = _card(referenced_fields=["不存在的指标"])
    issues = consistency_check(card, expr, _metrics(), reg)
    assert any("不是已知的原子指标" in i for i in issues), issues


def test_referenced_fields_suppresses_prose_scan(reg, op_names):
    """When the structural channel is populated, prose is not re-scanned —
    mentioning an unreferenced metric in prose is the LLM's declaration
    problem, caught only if it also lies in referenced_fields."""
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    card = _card(
        business_explanation="对比 净利润 口径时注意季节性。",
        referenced_fields=["营业收入"],
    )
    issues = consistency_check(card, expr, _metrics(), reg)
    assert issues == []


def test_prose_scan_masks_longer_names_first(reg, op_names):
    """Fallback scan: '营业收入同比增长率' in the tree+prose must not
    false-positive its substring '营业收入'."""
    expr = parse_sexpr("(GT 营业收入同比增长率 0.0)", op_names)
    card = _card(
        business_explanation="当 营业收入同比增长率 高于阈值时触发预警。",
    )
    issues = consistency_check(card, expr, _metrics(), reg)
    assert issues == [], issues


def test_threshold_direction_inversion_detected(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.5)", op_names)
    card = _card(
        business_explanation="当 营业收入 低于阈值且持续下降时触发。",
    )
    issues = consistency_check(card, expr, _metrics(), reg)
    assert any("方向" in i for i in issues), f"expected direction flag, got {issues}"


def test_quoted_statistics_drift_detected(reg, op_names):
    expr = parse_sexpr("(GT 营业收入 0.0)", op_names)
    card = _card(
        business_explanation="该指标 IV 0.999 表现优异。",
    )
    issues = consistency_check(card, expr, _metrics(iv=0.30), reg)
    assert any("偏差" in i for i in issues), f"expected stat drift flag, got {issues}"
