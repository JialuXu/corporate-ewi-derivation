"""Cross-batch LLM meta-analysis.

Reads a list of `ExplanationResult`s + optional `ExperimentRun`s from the
ledger, hands them to the LLM as a single prompt, and parses the response
into `MetaReport`. The LLM is asked to: (a) cluster cards by business theme,
(b) flag cross-batch redundancy, (c) identify coverage gaps, (d) self-check
against the five pitfalls in DESIGN.md §8.

On JSON parse or schema failure, `build` returns a partial MetaReport with
the raw output preserved, so the CLI can show it to the analyst and they can
re-prompt.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from auto_derivation.l4_explain.card import ExplanationResult
from auto_derivation.l4_explain.llm_client import LLMClient, parse_json_response

from .ledger import ExperimentRun
from .prompts import META_REPORT_SYSTEM_PROMPT, assemble_meta_report_prompt

logger = logging.getLogger(__name__)

PITFALL_KEYS = (
    "坑1_PSI单一稳定性",
    "坑2_征信NaN语义",
    "坑3_同期特征",
    "坑4_跨行业混搜",
    "坑5_LLM解释幻觉",
)


class Cluster(BaseModel):
    cluster_id: str
    business_theme: str
    member_card_indices: list[int] = Field(default_factory=list)
    representative_sexpr: str = ""


class RedundancyHit(BaseModel):
    card_index_a: int
    card_index_b: int
    similarity: float
    reasoning: str = ""


class MetaReport(BaseModel):
    semantic_clusters: list[Cluster] = Field(default_factory=list)
    cross_batch_redundancy: list[RedundancyHit] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    pitfall_self_check: dict[str, list[str]] = Field(default_factory=dict)
    overall_recommendation: str = ""
    raw_llm_output: str | None = None


def _cards_summary(
    results: Sequence[ExplanationResult],
    expr_sexprs: Sequence[str | None],
    industries: Sequence[str | None],
    fitness_list: Sequence[list[float] | None],
) -> list[dict[str, Any]]:
    """Compact dict per card for the prompt."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        c = r.card
        out.append({
            "index": i,
            "industry": industries[i] if i < len(industries) else None,
            "expr_sexpr": expr_sexprs[i] if i < len(expr_sexprs) else None,
            "metric_name_cn": c.metric_name_cn,
            "business_explanation": c.business_explanation,
            "use_cases": c.use_cases,
            "threshold_advice": c.threshold_advice,
            "fitness": fitness_list[i] if i < len(fitness_list) else None,
            "consistency_warnings": r.consistency_warnings,
        })
    return out


def _runs_summary(runs: Sequence[ExperimentRun]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": r.run_id,
            "label": r.label,
            "industry": r.industry,
            "seed": r.seed,
            "pareto_count": len(r.pareto_entries),
            "timestamp": r.timestamp.isoformat(),
            "panel_signature": r.panel_signature,
        }
        for r in runs
    ]


class MetaReportBuilder:
    """LLM-driven meta-analysis over a batch of explanation cards."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build(
        self,
        cards: Sequence[ExplanationResult],
        *,
        expr_sexprs: Sequence[str | None] | None = None,
        industries: Sequence[str | None] | None = None,
        fitness_list: Sequence[list[float] | None] | None = None,
        runs: Sequence[ExperimentRun] = (),
    ) -> MetaReport:
        n = len(cards)
        expr_sexprs_arg = list(expr_sexprs) if expr_sexprs is not None else [None] * n
        industries_arg = list(industries) if industries is not None else [None] * n
        fitness_arg = list(fitness_list) if fitness_list is not None else [None] * n

        cards_sum = _cards_summary(cards, expr_sexprs_arg, industries_arg, fitness_arg)
        runs_sum = _runs_summary(runs)
        industries_seen = sorted({i for i in industries_arg if i})
        labels_seen = sorted({r.label for r in runs})

        user = assemble_meta_report_prompt(
            cards_summary=cards_sum,
            runs_summary=runs_sum,
            industries_seen=industries_seen,
            labels_seen=labels_seen,
        )
        # The meta-report JSON (nested clusters + pitfalls) needs more room
        # than the client's 1024-token default.
        raw = self.llm.complete(
            system=META_REPORT_SYSTEM_PROMPT, user=user, max_tokens=4096,
        )

        try:
            parsed = parse_json_response(raw)
        except ValueError as e:
            logger.warning("meta_report: JSON parse failed: %s", e)
            return MetaReport(
                coverage_gaps=["llm_output_parse_failed"],
                overall_recommendation="LLM 输出无法解析为 JSON；见 raw_llm_output 字段并重试。",
                raw_llm_output=raw,
            )

        # Backfill missing pitfall keys with empty lists.
        pitfalls = parsed.get("pitfall_self_check") or {}
        if isinstance(pitfalls, dict):
            for k in PITFALL_KEYS:
                pitfalls.setdefault(k, [])
        parsed["pitfall_self_check"] = pitfalls

        try:
            report = MetaReport.model_validate(parsed)
        except ValidationError as e:
            logger.warning("meta_report: pydantic validation failed: %s", e)
            return MetaReport(
                coverage_gaps=["llm_output_schema_invalid"],
                overall_recommendation=f"LLM 输出 schema 不匹配: {e}",
                raw_llm_output=raw,
            )
        report.raw_llm_output = raw
        return report
