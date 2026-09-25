"""Tests for the meta-report builder."""
from __future__ import annotations

import json
from pathlib import Path

from auto_derivation.analysis.meta_report import (
    PITFALL_KEYS,
    MetaReport,
    MetaReportBuilder,
)
from auto_derivation.l4_explain.card import ExplanationCard, ExplanationResult
from auto_derivation.l4_explain.llm_client import MockClient


def _make_card(name: str) -> ExplanationResult:
    return ExplanationResult(
        card=ExplanationCard(
            metric_name_cn=name,
            business_explanation=f"{name} 的业务解释。",
            use_cases="制造业",
            diff_vs_existing="独特视角",
            threshold_advice="value > 0.5",
            review_checklist=["复核字段口径"],
        ),
        consistency_warnings=[],
    )


def _canned_meta_report() -> str:
    path = Path(__file__).parent / "fixtures" / "llm_responses" / "meta_report_canned.json"
    return path.read_text(encoding="utf-8")


def test_meta_report_builds_from_canned_response():
    client = MockClient(canned_json=_canned_meta_report())
    builder = MetaReportBuilder(llm=client)
    cards = [_make_card("营收同比下滑"), _make_card("营收环比下滑")]
    report = builder.build(
        cards,
        expr_sexprs=["(GT (PctChange 营业收入 12) -0.1)", "(GT (PctChange 营业收入 1) -0.05)"],
        industries=["制造", "制造"],
        fitness_list=[[0.3, 0.4, 0.8, 0.5, -3.0, 0.0], [0.28, 0.38, 0.78, 0.48, -3.0, 0.0]],
    )
    assert isinstance(report, MetaReport)
    assert len(report.semantic_clusters) == 1
    assert report.semantic_clusters[0].business_theme == "营收下滑预警"
    assert report.semantic_clusters[0].member_card_indices == [0, 1]
    assert len(report.cross_batch_redundancy) == 1
    assert report.cross_batch_redundancy[0].similarity == 0.85
    assert "建筑业" in report.coverage_gaps[0]
    # all pitfall keys present (even if empty)
    for k in PITFALL_KEYS:
        assert k in report.pitfall_self_check
    assert report.overall_recommendation.startswith("本批冗余")
    # raw_llm_output preserved
    assert report.raw_llm_output is not None


def test_meta_report_handles_missing_pitfall_keys():
    """LLM returns pitfall_self_check with only 2 of 5 keys — builder
    backfills the rest with empty lists so render code can iterate safely."""
    partial = {
        "semantic_clusters": [],
        "cross_batch_redundancy": [],
        "coverage_gaps": [],
        "pitfall_self_check": {"坑1_PSI单一稳定性": ["something"]},
        "overall_recommendation": "ok",
    }
    client = MockClient(canned_json=json.dumps(partial, ensure_ascii=False))
    report = MetaReportBuilder(llm=client).build([_make_card("x")])
    for k in PITFALL_KEYS:
        assert k in report.pitfall_self_check
    assert report.pitfall_self_check["坑1_PSI单一稳定性"] == ["something"]
    assert report.pitfall_self_check["坑5_LLM解释幻觉"] == []


def test_meta_report_parse_failure_is_recoverable():
    """If LLM returns non-JSON garbage, build() returns a MetaReport with
    raw_llm_output and a 'parse_failed' gap."""
    client = MockClient(canned_json="this is not json at all")
    report = MetaReportBuilder(llm=client).build([_make_card("x")])
    assert "llm_output_parse_failed" in report.coverage_gaps
    assert report.raw_llm_output == "this is not json at all"


def test_meta_report_schema_failure_is_recoverable():
    """Valid JSON but wrong types → schema error caught, raw preserved."""
    bad = json.dumps({"semantic_clusters": "not a list"})
    client = MockClient(canned_json=bad)
    report = MetaReportBuilder(llm=client).build([_make_card("x")])
    assert "llm_output_schema_invalid" in report.coverage_gaps
    assert report.raw_llm_output == bad


def test_meta_report_empty_cards_still_calls_llm():
    """Even with 0 cards (e.g. dry-run check), the builder should produce
    a MetaReport — the LLM may still return coverage gaps."""
    client = MockClient(canned_json=_canned_meta_report())
    report = MetaReportBuilder(llm=client).build([])
    assert isinstance(report, MetaReport)
    # client called once
    assert client.last_call is not None


def test_meta_report_prompt_includes_card_details():
    """Verify the prompt sent to the LLM mentions card content."""
    client = MockClient(canned_json=_canned_meta_report())
    cards = [_make_card("营收同比下滑")]
    MetaReportBuilder(llm=client).build(
        cards, industries=["制造"],
        expr_sexprs=["(GT (PctChange 营业收入 12) -0.1)"],
    )
    assert client.last_call is not None
    user_prompt = client.last_call["user"]
    assert "营收同比下滑" in user_prompt
    assert "PctChange" in user_prompt
    assert "制造" in user_prompt
