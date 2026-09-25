"""Tests for Markdown rendering of meta_report + suspicion."""
from __future__ import annotations

from datetime import datetime

from auto_derivation.analysis.ledger import ExperimentRun, ParetoEntry
from auto_derivation.analysis.meta_report import (
    Cluster,
    MetaReport,
    RedundancyHit,
)
from auto_derivation.analysis.render import (
    render_meta_report,
    render_suspicion_batch,
)
from auto_derivation.analysis.suspicion import SuspicionVerdict
from auto_derivation.l4_explain.card import ExplanationCard, ExplanationResult


def _card(name: str) -> ExplanationResult:
    return ExplanationResult(
        card=ExplanationCard(
            metric_name_cn=name,
            business_explanation="解释",
            use_cases="对公",
            diff_vs_existing="独特",
            threshold_advice="value > 0.5",
            review_checklist=["复核"],
        ),
    )


def _run() -> ExperimentRun:
    return ExperimentRun(
        run_id="abc12345",
        timestamp=datetime(2026, 5, 18, 10, 0),
        label="Y_npl_12m",
        industry="制造",
        seed=7,
        pareto_entries=[
            ParetoEntry(
                expr_json='{"kind":"field","name":"营业收入"}',
                expr_sexpr="营业收入",
                fitness=[0.3, 0.4, 0.8, 0.5, -3.0, 0.0],
                rank=0, crowding=1.0,
            )
        ],
    )


def test_render_meta_report_minimal():
    report = MetaReport(
        semantic_clusters=[
            Cluster(cluster_id="C1", business_theme="营收下滑",
                    member_card_indices=[0],
                    representative_sexpr="(GT 营业收入 0.5)"),
        ],
        cross_batch_redundancy=[
            RedundancyHit(card_index_a=0, card_index_b=1,
                          similarity=0.85, reasoning="阈值接近"),
        ],
        coverage_gaps=["建筑业 Y_overdue_3m 无候选"],
        pitfall_self_check={
            "坑1_PSI单一稳定性": [],
            "坑3_同期特征": ["卡片 0 需复核 lag"],
        },
        overall_recommendation="冗余高，建议补建筑业",
    )
    md = render_meta_report(
        report,
        runs=[_run()],
        cards=[_card("营收下滑"), _card("营收环比下滑")],
    )
    assert "# 跨批次元分析报告" in md
    assert "abc12345" in md
    assert "营收下滑" in md
    assert "建筑业 Y_overdue_3m 无候选" in md
    assert "卡片 0 需复核 lag" in md
    assert "冗余高，建议补建筑业" in md
    # cluster members should be looked up
    assert "[0]" in md  # member index


def test_render_meta_report_with_parse_failure_includes_raw():
    report = MetaReport(
        coverage_gaps=["llm_output_parse_failed"],
        overall_recommendation="解析失败",
        raw_llm_output="garbage from LLM",
    )
    md = render_meta_report(report)
    assert "raw_llm_output" in md
    assert "garbage from LLM" in md


def test_render_suspicion_batch_sorts_by_composite():
    cards = [_card("规则A"), _card("规则B"), _card("规则C")]
    verdicts = [
        SuspicionVerdict(industry_fit=0.1, drift_risk=0.2, spurious_risk=0.1,
                         reasoning="安全", flags=["industry_consistent"]),
        SuspicionVerdict(industry_fit=0.9, drift_risk=0.3, spurious_risk=0.5,
                         reasoning="高 industry_fit 不匹配", flags=["high_risk"]),
        SuspicionVerdict(industry_fit=0.4, drift_risk=0.4, spurious_risk=0.8,
                         reasoning="可能伪相关", flags=["spurious_suspect"]),
    ]
    md = render_suspicion_batch(
        verdicts, cards,
        expr_sexprs=["A", "B", "C"],
        industries=["制造", "批零", "建筑"],
    )
    # Composite ordering: B(0.9), C(0.8), A(0.2)
    # The detail headers should appear in that order.
    idx_b = md.index("规则B")
    idx_c = md.index("规则C")
    idx_a = md.index("规则A")
    assert idx_b < idx_c < idx_a
    # Markdown table should have rows for all 3
    assert "高 industry_fit 不匹配" in md
    assert "industry_consistent" in md


def test_render_suspicion_batch_empty():
    md = render_suspicion_batch([], [], [], [])
    assert "共 0 条" in md
