"""Prompt for cross-batch meta-analysis of explanation cards."""
from __future__ import annotations

import json
from typing import Any

META_REPORT_SYSTEM_PROMPT = (
    "你是对公贷后预警的高级风控分析师。给你一批模型自动衍生的候选预警规则"
    "（每条含 S-表达式、中文业务解释、IV/KS/PSI 等指标、所属行业），"
    "你要做四件事："
    "(1) 语义聚类——把讲同一件事的规则归到一起；"
    "(2) 跨批次冗余识别——指出不同切片下重复发现的规则模式；"
    "(3) 覆盖 gap 识别——哪些 (行业, 标签) 切片缺少有效候选；"
    "(4) 五类陷阱自检——逐条对照五类陷阱，标出嫌疑候选。"
    "严格按 JSON schema 输出，不要任何 markdown 包裹、不要解释、不要多余文本。"
)


def assemble_meta_report_prompt(
    *,
    cards_summary: list[dict[str, Any]],
    runs_summary: list[dict[str, Any]],
    industries_seen: list[str],
    labels_seen: list[str],
) -> str:
    """Build the user-prompt body.

    `cards_summary` is a compact list — each element has card fields plus
    expr_sexpr, fitness (6 floats), industry. We send the JSON directly so
    the LLM can quote chunk_ids back at us in evidence/cluster references.
    """
    cards_json = json.dumps(cards_summary, ensure_ascii=False, indent=2)
    runs_json = json.dumps(runs_summary, ensure_ascii=False, indent=2)
    industries_str = "、".join(industries_seen) or "（无）"
    labels_str = "、".join(labels_seen) or "（无）"

    schema_example = """{
  "semantic_clusters": [
    {
      "cluster_id": "C1",
      "business_theme": "营收下滑 + 资金流恶化",
      "member_card_indices": [0, 3, 7],
      "representative_sexpr": "(GT (PctChange 营业收入 1) -0.3)"
    }
  ],
  "cross_batch_redundancy": [
    {
      "card_index_a": 2,
      "card_index_b": 5,
      "similarity": 0.85,
      "reasoning": "都是基于资产负债率单变量阈值，行业切片不同但本质同一规则"
    }
  ],
  "coverage_gaps": [
    "建筑业的 Y_overdue_3m 标签下无有效候选",
    "Y_npl_12m 在批零业的非支配候选少于 3 个"
  ],
  "pitfall_self_check": {
    "坑1_PSI单一稳定性": ["卡片 0、3 的 PSI 在 0.1-0.25 之间，KL 散度未校验"],
    "坑2_征信NaN语义": [],
    "坑3_同期特征": ["卡片 4 看起来用了同期触发字段，需复核 lag"],
    "坑4_跨行业混搜": [],
    "坑5_LLM解释幻觉": ["卡片 6 的业务解释中提到了表达式树未引用的字段"]
  },
  "overall_recommendation": "本批整体偏冗余，建议在批零和建筑业切片扩大探索。"
}"""

    return f"""【本批候选卡片（共 {len(cards_summary)} 条）】
{cards_json}

【本批 GP 运行（共 {len(runs_summary)} 次）】
{runs_json}

【已覆盖切片】
- 行业: {industries_str}
- 标签: {labels_str}

【任务】严格按以下 JSON schema 输出（字段名照搬，不增不减）：
{schema_example}

约束：
- `member_card_indices` 必须是上面卡片列表的 0 起始 index
- `pitfall_self_check` 的 5 个键名必须完整（即使值是空 list）
- 不要写 markdown 代码块包裹 JSON
- 不要在 JSON 前后添加任何解释文字
"""
