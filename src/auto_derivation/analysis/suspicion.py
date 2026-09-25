"""Per-card suspicion scoring with optional industry-RAG evidence.

For each (card, expr, industry, fitness) tuple, ask the LLM for three
[0,1] risk scores (industry_fit, drift_risk, spurious_risk) plus reasoning
and a flags array. If a `TfidfRetriever` is supplied, retrieve top-k
industry knowledge chunks first and inject them as evidence in the prompt.

When the retriever is None or empty (cold-start corpus), we degrade
gracefully: still call the LLM but with no evidence block, and add
`flag="no_corpus"` to the verdict so downstream tools can highlight that
this judgement is unanchored.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from auto_derivation.expression.tree import ExprNode
from auto_derivation.knowledge.retriever import TfidfRetriever
from auto_derivation.l1_data.registry import MetricRegistry
from auto_derivation.l4_explain.card import ExplanationResult
from auto_derivation.l4_explain.llm_client import LLMClient, parse_json_response

from .prompts import SUSPICION_SYSTEM_PROMPT, assemble_suspicion_prompt

logger = logging.getLogger(__name__)


class SuspicionVerdict(BaseModel):
    industry_fit: float = Field(ge=0.0, le=1.0)
    drift_risk: float = Field(ge=0.0, le=1.0)
    spurious_risk: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    evidence_doc_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    @property
    def composite(self) -> float:
        """One aggregate score for sorting: max of the three. Higher = more suspicious."""
        return max(self.industry_fit, self.drift_risk, self.spurious_risk)


class SuspicionInput(BaseModel):
    """One scoring request, packaged for `SuspicionScorer.score_batch`."""

    explanation: ExplanationResult
    expr: ExprNode
    industry: str | None = None
    fitness: list[float] | None = None

    model_config = {"arbitrary_types_allowed": True}


def _build_query_for_retriever(card_payload: dict[str, Any], expr_sexpr: str) -> str:
    """Concatenate the key card text fields into a search query."""
    parts = [
        card_payload.get("metric_name_cn", ""),
        card_payload.get("business_explanation", ""),
        card_payload.get("threshold_advice", ""),
        expr_sexpr,
    ]
    return " ".join(p for p in parts if p)


class SuspicionScorer:
    """LLM-backed suspicion scoring, optionally grounded in industry RAG.

    The scorer never raises on per-card failures — it logs and returns a
    fallback verdict with `flags=["llm_failed"]` so a batch doesn't lose
    everything when one card explodes.
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: TfidfRetriever | None = None,
        registry: MetricRegistry | None = None,
        *,
        top_k: int = 5,
    ):
        self.llm = llm
        self.retriever = retriever
        self.registry = registry
        self.top_k = top_k

    def _retrieve(
        self, query: str, industry: str | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (evidence_snippets, flags). flags signals 'no_corpus' when applicable."""
        if self.retriever is None or len(self.retriever) == 0:
            return [], ["no_corpus"]
        hits = self.retriever.query(query, industry=industry, top_k=self.top_k)
        if not hits:
            return [], ["no_evidence_match"]
        snippets = [
            {"chunk_id": c.chunk_id, "text": c.text[:400], "score": round(score, 3)}
            for c, score in hits
        ]
        return snippets, []

    def score(
        self,
        explanation: ExplanationResult,
        expr: ExprNode,
        industry: str | None = None,
        fitness: list[float] | None = None,
    ) -> SuspicionVerdict:
        card_payload = explanation.card.model_dump()
        expr_sexpr = expr.to_sexpr()

        query = _build_query_for_retriever(card_payload, expr_sexpr)
        evidence, retrieval_flags = self._retrieve(query, industry)

        user = assemble_suspicion_prompt(
            card_payload=card_payload,
            expr_sexpr=expr_sexpr,
            industry=industry,
            fitness=fitness,
            evidence_snippets=evidence,
        )

        try:
            raw = self.llm.complete(
                system=SUSPICION_SYSTEM_PROMPT, user=user, max_tokens=1024,
            )
        except Exception as e:
            logger.warning("suspicion: LLM call failed: %s", e)
            return SuspicionVerdict(
                industry_fit=0.5, drift_risk=0.5, spurious_risk=0.5,
                reasoning=f"LLM 调用失败: {e}",
                evidence_doc_ids=[s["chunk_id"] for s in evidence],
                flags=[*retrieval_flags, "llm_failed"],
            )

        try:
            parsed = parse_json_response(raw)
        except ValueError as e:
            logger.warning("suspicion: JSON parse failed: %s", e)
            return SuspicionVerdict(
                industry_fit=0.5, drift_risk=0.5, spurious_risk=0.5,
                reasoning=f"LLM 输出无法解析为 JSON: {raw[:200]!r}",
                evidence_doc_ids=[s["chunk_id"] for s in evidence],
                flags=[*retrieval_flags, "llm_parse_failed"],
            )

        try:
            verdict = SuspicionVerdict.model_validate(parsed)
        except ValidationError as e:
            logger.warning("suspicion: pydantic validation failed: %s", e)
            return SuspicionVerdict(
                industry_fit=0.5, drift_risk=0.5, spurious_risk=0.5,
                reasoning=f"schema 不匹配: {e}",
                evidence_doc_ids=[s["chunk_id"] for s in evidence],
                flags=[*retrieval_flags, "llm_schema_invalid"],
            )

        # Merge retrieval-stage flags into verdict.flags (de-duplicated).
        if retrieval_flags:
            seen = set(verdict.flags)
            for f in retrieval_flags:
                if f not in seen:
                    verdict.flags.append(f)
                    seen.add(f)
        return verdict

    def score_batch(
        self,
        items: Sequence[SuspicionInput],
    ) -> list[SuspicionVerdict]:
        """Sequential batch scoring: the LLM API has rate limits and the openai
        SDK isn't safely shared across threads in all versions. The CLI's
        `--limit` flag caps the batch size; run multiple batches manually."""
        out: list[SuspicionVerdict] = []
        for item in items:
            out.append(self.score(
                explanation=item.explanation, expr=item.expr,
                industry=item.industry, fitness=item.fitness,
            ))
        return out
