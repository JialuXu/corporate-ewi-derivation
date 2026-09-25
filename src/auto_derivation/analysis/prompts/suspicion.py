"""Prompt for per-card suspicion scoring."""
from __future__ import annotations

import json
from typing import Any

SUSPICION_SYSTEM_PROMPT = (
    "你是对公贷后预警的资深风控审核员。给你一条模型自动衍生的候选预警规则"
    "（含 S-表达式、业务解释、行业、IV/KS/PSI 等数值，以及可选的行业知识片段），"
    "你要给出三个 [0,1] 区间的可疑度评分："
    "(a) industry_fit — 在所属行业里讲不通的程度（0=合理，1=完全不合行业逻辑）；"
    "(b) drift_risk — 受概念漂移影响的可能性（启发式判断）；"
    "(c) spurious_risk — 是统计伪相关而非因果信号的可能性。"
    "还要给出 reasoning（中文，120 字以内）解释三个分数的来源，"
    "以及 flags（关键词数组）。"
    "只输出符合 schema 的 JSON 对象本身。"
)


def assemble_suspicion_prompt(
    *,
    card_payload: dict[str, Any],
    expr_sexpr: str,
    industry: str | None,
    fitness: list[float] | None,
    evidence_snippets: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user-prompt body for one card."""
    card_json = json.dumps(card_payload, ensure_ascii=False, indent=2)
    industry_str = industry or "（未指定）"
    fitness_str = (
        json.dumps(
            {
                "IV": fitness[0] if fitness and len(fitness) > 0 else None,
                "KS": fitness[1] if fitness and len(fitness) > 1 else None,
                "stability": fitness[2] if fitness and len(fitness) > 2 else None,
                "monotonicity": fitness[3] if fitness and len(fitness) > 3 else None,
                "leaves": int(-fitness[4]) if fitness and len(fitness) > 4 else None,
                "redundancy": -fitness[5] if fitness and len(fitness) > 5 else None,
            },
            ensure_ascii=False,
        )
        if fitness
        else "（无）"
    )

    evidence_block: str
    if evidence_snippets:
        evidence_block = "\n".join(
            f"- [{e['chunk_id']}] {e['text']}"
            for e in evidence_snippets
        )
    else:
        evidence_block = "（无行业知识片段；按通用常识判断）"

    schema_example = """{
  "industry_fit": 0.30,
  "drift_risk": 0.10,
  "spurious_risk": 0.20,
  "reasoning": "营收下滑+资产负债率上升在制造业是经典预警信号，行业知识支持，统计稳定。",
  "evidence_doc_ids": ["industries/制造#应收账款的健康水位"],
  "flags": ["industry_consistent"]
}"""

    return f"""【候选规则】
- 行业: {industry_str}
- S-表达式: {expr_sexpr}
- 业务卡片:
{card_json}

【数值表现】
{fitness_str}

【行业知识片段（来自知识库 RAG）】
{evidence_block}

【任务】严格按以下 JSON schema 输出：
{schema_example}

约束：
- 三个评分都是 [0, 1] 之间的浮点数
- evidence_doc_ids 必须来自上面给出的片段 chunk_id
- flags 是简短关键词数组，例如 ["industry_consistent", "high_iv", "leakage_suspect"]
- reasoning 中文，不超过 120 字
"""
