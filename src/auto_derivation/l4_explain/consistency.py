"""Machine consistency check (DESIGN.md §6.2).

Catches the most common LLM hallucinations:

1. **Field hallucination**: business_explanation mentions a field name that
   does not appear in the actual expression tree.
2. **Threshold direction mismatch**: the tree uses `GT(x, t)` but the card
   says "...低于阈值时触发..." — direction inverted.
3. **Statistic drift**: the card cites IV/KS that diverge sharply from what
   we measured (>10% relative). Optional, only if numbers appear in text.
"""
from __future__ import annotations

import re

from auto_derivation.evaluation.metrics import DiscriminationMetrics
from auto_derivation.expression.evaluator import collect_field_names
from auto_derivation.expression.tree import ExprNode, OpNode
from auto_derivation.l1_data.registry import MetricRegistry

from .card import ExplanationCard


def consistency_check(
    card: ExplanationCard,
    expr: ExprNode,
    metrics: DiscriminationMetrics,
    registry: MetricRegistry,
) -> list[str]:
    issues: list[str] = []
    issues.extend(_check_fields(card, expr, registry))
    issues.extend(_check_threshold_direction(card, expr))
    issues.extend(_check_statistics(card, metrics))
    return issues


def _check_fields(
    card: ExplanationCard, expr: ExprNode, registry: MetricRegistry
) -> list[str]:
    """Every metric the explanation references must appear in the tree.

    Primary channel: `card.referenced_fields` (the LLM declares its references
    explicitly — structural, no substring ambiguity). Fallback when the card
    leaves it empty: scan the free-form prose for registry names, longest
    first, masking each match so "营业收入" can't false-positive inside
    "营业收入同比增长率"."""
    tree_field_names: set[str] = set()
    for raw in collect_field_names(expr):
        try:
            meta = registry.resolve(raw)
        except KeyError:
            continue
        tree_field_names.add(meta.name_cn)
        tree_field_names.add(meta.metric_id)

    if card.referenced_fields:
        return _check_declared_fields(card.referenced_fields, tree_field_names, registry)
    return _scan_prose(card.business_explanation, tree_field_names, registry)


def _check_declared_fields(
    declared: list[str], tree_field_names: set[str], registry: MetricRegistry
) -> list[str]:
    issues: list[str] = []
    for name in declared:
        if name in tree_field_names:
            continue
        try:
            meta = registry.resolve(name)
        except KeyError:
            issues.append(
                f"referenced_fields 包含 '{name}'，但它不是已知的原子指标"
            )
            continue
        if meta.name_cn not in tree_field_names and meta.metric_id not in tree_field_names:
            issues.append(
                f"业务解释声明引用字段 '{name}'，但表达式树并未引用它"
            )
    return issues


def _scan_prose(
    text: str, tree_field_names: set[str], registry: MetricRegistry
) -> list[str]:
    issues: list[str] = []
    masked = text
    for name in registry.names_cn_by_length_desc():
        if name not in masked:
            continue
        masked = masked.replace(name, "\x00" * len(name))
        if name not in tree_field_names:
            issues.append(
                f"业务解释提到字段 '{name}'，但表达式树并未引用它"
            )
    return issues


_GT_DOWN_PATTERNS = ("低于", "下降", "下滑", "回落", "减少")
_LT_UP_PATTERNS = ("高于", "上升", "上涨", "增加", "激增")


def _check_threshold_direction(card: ExplanationCard, expr: ExprNode) -> list[str]:
    """Heuristic: collect threshold ops in the tree; for each GT(x, t) the
    explanation should not say 'x 低于阈值时触发', and vice versa."""
    has_gt = _tree_has_op(expr, "GT")
    has_lt = _tree_has_op(expr, "LT")
    text = card.business_explanation + " " + card.threshold_advice

    issues: list[str] = []
    if has_gt and not has_lt and any(p in text for p in _GT_DOWN_PATTERNS):
        # GT triggers when value is *high*, but text says "low" — possibly inverted.
        # Only warn when GT is the only directional op (avoid noise on AND of opposites).
        issues.append(
            "表达式使用 GT(value > threshold)，但解释包含 '低于/下降' 类词，方向可能反了"
        )
    if has_lt and not has_gt and any(p in text for p in _LT_UP_PATTERNS):
        issues.append(
            "表达式使用 LT(value < threshold)，但解释包含 '高于/上升' 类词，方向可能反了"
        )
    return issues


def _tree_has_op(node: ExprNode, op_name: str) -> bool:
    if isinstance(node, OpNode):
        if node.op_name == op_name:
            return True
        return any(_tree_has_op(c, op_name) for c in node.children)
    return False


_NUM_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
_NUM_RAW_RE = re.compile(r"IV[^0-9]{0,3}(\d+\.\d+)|KS[^0-9]{0,3}(\d+\.\d+)")


def _check_statistics(card: ExplanationCard, metrics: DiscriminationMetrics) -> list[str]:
    """If the LLM quotes IV/KS in its narrative, they must be within 10% of
    the measured values. We don't fail when no numbers are quoted."""
    issues: list[str] = []
    text = card.business_explanation + " " + card.diff_vs_existing
    for m in _NUM_RAW_RE.finditer(text):
        for grp in m.groups():
            if grp is None:
                continue
            quoted = float(grp)
            # Match against IV or KS depending on prefix in the matched span.
            ref = metrics.iv if "IV" in m.group(0).upper() else metrics.ks
            if ref == 0:
                continue
            if abs(quoted - ref) / max(abs(ref), 1e-6) > 0.1:
                issues.append(
                    f"解释中引用 {m.group(0)}，与实测值 {ref:.3f} 偏差 > 10%"
                )
    return issues
