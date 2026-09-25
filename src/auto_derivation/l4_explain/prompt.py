"""Prompt assembly for L4 explanation.

Mirrors the structured prompt in DESIGN.md §6.1: tree + field metadata +
known statistics → instruction asking for the six card fields in JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

from auto_derivation.evaluation.metrics import DiscriminationMetrics
from auto_derivation.expression.evaluator import collect_field_names
from auto_derivation.expression.tree import ExprNode
from auto_derivation.l1_data.registry import MetricRegistry


@dataclass
class PromptContext:
    expr: ExprNode
    metrics: DiscriminationMetrics
    registry: MetricRegistry
    similar_metric_name: str | None = None  # name of nearest existing metric, if any
    similar_correlation: float | None = None


SYSTEM_PROMPT = (
    "你是对公信贷风险专家，专攻贷后预警。你把模型自动衍生出的复合指标"
    "翻译成业务人员能直接读懂的卡片。"
    "卡片必须用中文，字段含义见 schema。"
    "严格按 JSON 输出，不要任何 markdown 包裹、不要解释、不要多余文本。"
)


def assemble_user_prompt(ctx: PromptContext) -> str:
    field_lines = []
    for fname in collect_field_names(ctx.expr):
        try:
            meta = ctx.registry.resolve(fname)
        except KeyError:
            continue
        field_lines.append(
            f"- {meta.name_cn}（{meta.metric_id}）："
            f"dtype={meta.dtype.value}，频率={meta.time_grain.value}，"
            f"取值范围={meta.value_range or '未声明'}，"
            f"业务标签={meta.business_tag or '无'}"
        )
    fields_block = "\n".join(field_lines) or "（无可解析字段）"

    similarity_block = (
        f"\n- 与已有相似指标 '{ctx.similar_metric_name}' 的相关性: "
        f"{ctx.similar_correlation:.2f}"
        if ctx.similar_metric_name and ctx.similar_correlation is not None
        else ""
    )

    trigger_line = (
        f"\n- 触发率: {ctx.metrics.trigger_rate:.2%}"
        if ctx.metrics.trigger_rate is not None
        else ""
    )

    return f"""【表达式树（S-表达式）】
{ctx.expr.to_sexpr()}

【字段元信息】
{fields_block}

【已知统计表现】
- IV: {ctx.metrics.iv:.3f}
- KS: {ctx.metrics.ks:.3f}
- Gini: {ctx.metrics.gini:.3f}
- 样本数: {ctx.metrics.n}（其中正样本 {ctx.metrics.n_pos}）{trigger_line}{similarity_block}

【任务】
请输出 JSON，对应字段含义：
{{
  "metric_name_cn": "中文指标名，必须 ≤15 字，体现业务含义",
  "business_explanation": "2-3 句话，讲明触发逻辑和风险含义",
  "use_cases": "适用客户类型/时点",
  "diff_vs_existing": "与已有指标的差异点（如无相似指标可写出本指标的独特视角）",
  "threshold_advice": "建议的预警阈值范围",
  "review_checklist": ["复核要点1", "复核要点2", ...],
  "referenced_fields": ["解释中实际引用的原子指标中文名", ...]
}}

约束：
- metric_name_cn 必须 ≤15 字符
- business_explanation 中提到的所有字段必须真实出现在表达式树里
- referenced_fields 只能列【字段元信息】中出现的中文名，且解释正文提到的每个字段都要列入
- 阈值方向必须与表达式中的 GT/LT 方向一致
"""
