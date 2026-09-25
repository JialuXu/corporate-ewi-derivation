"""Tests for the suspicion scorer."""
from __future__ import annotations

import json
from pathlib import Path

from auto_derivation.analysis.suspicion import (
    SuspicionInput,
    SuspicionScorer,
    SuspicionVerdict,
)
from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.knowledge.loader import load_corpus
from auto_derivation.knowledge.retriever import TfidfRetriever
from auto_derivation.l2_operators.registry import default_operator_registry
from auto_derivation.l4_explain.card import ExplanationCard, ExplanationResult
from auto_derivation.l4_explain.llm_client import MockClient


def _make_card() -> ExplanationResult:
    return ExplanationResult(
        card=ExplanationCard(
            metric_name_cn="营收同比下滑预警",
            business_explanation="当营业收入同比下滑超过 10% 时触发，提示制造业经营弱化。",
            use_cases="制造业对公客户",
            diff_vs_existing="对短周期下滑更敏感",
            threshold_advice="PctChange < -0.1 时强预警",
            review_checklist=["核查营收口径", "比对资金流"],
        ),
        consistency_warnings=[],
    )


def _canned_suspicion() -> str:
    path = Path(__file__).parent / "fixtures" / "llm_responses" / "suspicion_canned.json"
    return path.read_text(encoding="utf-8")


def _expr():
    op_names = set(default_operator_registry().names())
    return parse_sexpr("(GT (PctChange 营业收入 12) -0.1)", op_names)


def test_suspicion_score_with_corpus(tmp_path):
    """Suspicion + corpus → evidence_doc_ids populated, no_corpus flag absent."""
    docs = load_corpus(Path(__file__).parent / "fixtures" / "mini_corpus")
    chunks = [c for d in docs for c in d.chunks]
    retriever = TfidfRetriever(chunks)
    retriever.fit()

    scorer = SuspicionScorer(
        llm=MockClient(canned_json=_canned_suspicion()),
        retriever=retriever,
    )
    verdict = scorer.score(
        _make_card(), _expr(), industry="制造",
        fitness=[0.3, 0.4, 0.8, 0.5, -3.0, 0.0],
    )
    assert isinstance(verdict, SuspicionVerdict)
    assert 0.0 <= verdict.industry_fit <= 1.0
    assert 0.0 <= verdict.drift_risk <= 1.0
    assert 0.0 <= verdict.spurious_risk <= 1.0
    assert "industry_consistent" in verdict.flags
    assert "no_corpus" not in verdict.flags
    assert verdict.evidence_doc_ids   # canned response has one


def test_suspicion_cold_start_no_corpus():
    """retriever=None → verdict flags include 'no_corpus'."""
    scorer = SuspicionScorer(
        llm=MockClient(canned_json=_canned_suspicion()),
        retriever=None,
    )
    verdict = scorer.score(_make_card(), _expr(), industry="制造")
    assert "no_corpus" in verdict.flags


def test_suspicion_empty_corpus_flag():
    """retriever fit on zero chunks → still no_corpus flag."""
    retriever = TfidfRetriever([])
    retriever.fit()
    scorer = SuspicionScorer(
        llm=MockClient(canned_json=_canned_suspicion()),
        retriever=retriever,
    )
    verdict = scorer.score(_make_card(), _expr(), industry="制造")
    assert "no_corpus" in verdict.flags


def test_suspicion_llm_parse_failure_returns_fallback():
    """Bad LLM output → fallback verdict with 0.5 mid-scores + parse_failed flag."""
    scorer = SuspicionScorer(
        llm=MockClient(canned_json="garbage not json"),
        retriever=None,
    )
    verdict = scorer.score(_make_card(), _expr())
    assert verdict.industry_fit == 0.5
    assert "llm_parse_failed" in verdict.flags


def test_suspicion_llm_schema_failure_returns_fallback():
    """Valid JSON, invalid schema → fallback + llm_schema_invalid flag."""
    scorer = SuspicionScorer(
        llm=MockClient(canned_json=json.dumps({"industry_fit": "not_a_float"})),
        retriever=None,
    )
    verdict = scorer.score(_make_card(), _expr())
    assert verdict.industry_fit == 0.5
    assert "llm_schema_invalid" in verdict.flags


def test_suspicion_composite_score():
    v = SuspicionVerdict(
        industry_fit=0.3, drift_risk=0.7, spurious_risk=0.5, reasoning="…",
    )
    assert v.composite == 0.7


def test_suspicion_score_batch():
    """Batch returns one verdict per input, in order."""
    scorer = SuspicionScorer(
        llm=MockClient(canned_json=_canned_suspicion()),
        retriever=None,
    )
    items = [
        SuspicionInput(explanation=_make_card(), expr=_expr(), industry="制造",
                       fitness=[0.3, 0.4, 0.8, 0.5, -3.0, 0.0]),
        SuspicionInput(explanation=_make_card(), expr=_expr(), industry="批零",
                       fitness=[0.2, 0.3, 0.7, 0.4, -4.0, -0.1]),
    ]
    verdicts = scorer.score_batch(items)
    assert len(verdicts) == 2
    assert all(isinstance(v, SuspicionVerdict) for v in verdicts)
