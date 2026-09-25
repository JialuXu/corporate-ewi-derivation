"""Markdown rendering for human analysts.

Three flavours: meta-report, suspicion batch, frontier (frontier already
lives in `analysis.frontier`; re-exported here for convenience).
"""
from __future__ import annotations

from collections.abc import Sequence

from auto_derivation.l4_explain.card import ExplanationResult

from .frontier import render_frontier  # re-export
from .ledger import ExperimentRun
from .meta_report import MetaReport
from .suspicion import SuspicionVerdict

__all__ = ["render_frontier", "render_meta_report", "render_suspicion_batch"]


def render_meta_report(
    report: MetaReport,
    runs: Sequence[ExperimentRun] = (),
    cards: Sequence[ExplanationResult] = (),
) -> str:
    """Render a MetaReport as a Markdown document for analyst review."""
    lines: list[str] = ["# 跨批次元分析报告", ""]

    if runs:
        lines.append(f"覆盖 {len(runs)} 次 GP 运行 / {len(cards)} 张候选卡片")
        lines.append("")
        lines.append("| run_id | label | industry | seed | pareto_count | timestamp |")
        lines.append("|--------|-------|----------|------|--------------|-----------|")
        for r in runs:
            lines.append(
                f"| {r.run_id} | {r.label} | {r.industry or '-'} | {r.seed} | "
                f"{len(r.pareto_entries)} | {r.timestamp.strftime('%Y-%m-%d %H:%M')} |"
            )
        lines.append("")

    # Coverage gaps
    lines.append("## 覆盖 gap")
    if report.coverage_gaps:
        for g in report.coverage_gaps:
            lines.append(f"- {g}")
    else:
        lines.append("（无）")
    lines.append("")

    # Semantic clusters
    lines.append("## 语义聚类")
    if not report.semantic_clusters:
        lines.append("（无）")
    else:
        for cl in report.semantic_clusters:
            lines.append(f"### {cl.cluster_id} — {cl.business_theme}")
            lines.append(f"- 代表表达式: `{cl.representative_sexpr}`")
            members = ", ".join(str(i) for i in cl.member_card_indices)
            lines.append(f"- 成员卡片 index: {members or '(无)'}")
            if cards and cl.member_card_indices:
                lines.append("- 成员摘要:")
                for i in cl.member_card_indices:
                    if 0 <= i < len(cards):
                        name = cards[i].card.metric_name_cn
                        lines.append(f"  - [{i}] {name}")
            lines.append("")

    # Cross-batch redundancy
    lines.append("## 跨批次冗余")
    if not report.cross_batch_redundancy:
        lines.append("（无）")
    else:
        lines.append("| 卡片 A | 卡片 B | 相似度 | 说明 |")
        lines.append("|--------|--------|--------|------|")
        for h in report.cross_batch_redundancy:
            name_a = (
                cards[h.card_index_a].card.metric_name_cn
                if cards and 0 <= h.card_index_a < len(cards) else f"#{h.card_index_a}"
            )
            name_b = (
                cards[h.card_index_b].card.metric_name_cn
                if cards and 0 <= h.card_index_b < len(cards) else f"#{h.card_index_b}"
            )
            lines.append(f"| {name_a} | {name_b} | {h.similarity:.2f} | {h.reasoning} |")
    lines.append("")

    # Pitfall self-check
    lines.append("## 五类陷阱自检")
    if not report.pitfall_self_check:
        lines.append("（未输出，可能 LLM 解析失败，参见 raw_llm_output）")
    else:
        for key, items in report.pitfall_self_check.items():
            lines.append(f"### {key}")
            if items:
                for it in items:
                    lines.append(f"- {it}")
            else:
                lines.append("- （无嫌疑）")
            lines.append("")

    # Overall
    lines.append("## 总评")
    lines.append(report.overall_recommendation or "（无）")
    lines.append("")

    if report.raw_llm_output and ("llm_output_parse_failed" in report.coverage_gaps
                                   or "llm_output_schema_invalid" in report.coverage_gaps):
        lines.append("## raw_llm_output（解析失败时的原始 LLM 输出）")
        lines.append("```")
        lines.append(report.raw_llm_output)
        lines.append("```")
    return "\n".join(lines)


def render_suspicion_batch(
    verdicts: Sequence[SuspicionVerdict],
    cards: Sequence[ExplanationResult],
    expr_sexprs: Sequence[str] = (),
    industries: Sequence[str | None] = (),
) -> str:
    """Render a list of suspicion verdicts, sorted by composite score desc."""
    indexed = list(enumerate(verdicts))
    indexed.sort(key=lambda pair: -pair[1].composite)
    lines: list[str] = ["# 候选指标可疑度评分", ""]
    lines.append(f"共 {len(verdicts)} 条；按 max(industry_fit, drift_risk, spurious_risk) 降序")
    lines.append("")
    lines.append("## 排序表")
    lines.append("| # | 指标名 | 行业 | industry_fit | drift_risk | spurious_risk | flags |")
    lines.append("|---|--------|------|--------------|------------|---------------|-------|")
    for orig_idx, v in indexed:
        name = (
            cards[orig_idx].card.metric_name_cn
            if 0 <= orig_idx < len(cards) else f"#{orig_idx}"
        )
        ind = industries[orig_idx] if industries and 0 <= orig_idx < len(industries) else "-"
        flags_str = ", ".join(v.flags) if v.flags else "-"
        lines.append(
            f"| {orig_idx} | {name} | {ind or '-'} | "
            f"{v.industry_fit:.2f} | {v.drift_risk:.2f} | {v.spurious_risk:.2f} | {flags_str} |"
        )
    lines.append("")
    lines.append("## 详情")
    for orig_idx, v in indexed:
        name = (
            cards[orig_idx].card.metric_name_cn
            if 0 <= orig_idx < len(cards) else f"#{orig_idx}"
        )
        expr_str = (
            expr_sexprs[orig_idx]
            if expr_sexprs and 0 <= orig_idx < len(expr_sexprs) else "-"
        )
        lines.append(f"### #{orig_idx} {name}")
        lines.append(f"- 表达式: `{expr_str}`")
        lines.append(
            f"- 评分: industry_fit={v.industry_fit:.2f}, "
            f"drift_risk={v.drift_risk:.2f}, spurious_risk={v.spurious_risk:.2f}"
        )
        lines.append(f"- flags: {', '.join(v.flags) if v.flags else '（无）'}")
        if v.evidence_doc_ids:
            lines.append(f"- 引用知识: {', '.join(v.evidence_doc_ids)}")
        lines.append(f"- 推理: {v.reasoning}")
        lines.append("")
    return "\n".join(lines)
