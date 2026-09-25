"""Real LLM coach (DESIGN.md §5.3) backed by any OpenAI-compatible endpoint.

Implements the `LLMCoach` Protocol from `llm_coach.py` by reusing the L4
transport (`OpenAICompatClient`) — we don't fork the LLM plumbing between
L3 and L4.

Every method degrades gracefully: a coach failure (network error, bad JSON,
unparsable S-expression) must never kill the GP search, so errors are logged
and neutral defaults returned (empty seed list / 0.5 critique scores).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from auto_derivation.expression.tree import ExprNode, parse_sexpr
from auto_derivation.l4_explain.llm_client import (
    LLMClient,
    OpenAICompatClient,
    parse_json_response,
)

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是对公贷后预警领域的风控专家，辅助一个遗传规划（GP）搜索引擎衍生新的预警指标。"
    "表达式使用 S-表达式语法，例如 (GT (PctChange 营业收入 4) 0.3)。"
    "只能使用提供的算子名和字段名，不得编造。"
    "严格按 JSON 输出，不要 markdown 包裹、不要解释、不要多余文本。"
)

# Keep prompts bounded — the registry has ~800 fields; a sample is enough
# context for the coach to propose plausible combinations.
_MAX_FIELDS_IN_PROMPT = 120


def _fields_block(names: list[str]) -> str:
    shown = names[:_MAX_FIELDS_IN_PROMPT]
    suffix = ""
    if len(names) > len(shown):
        suffix = f"\n…（共 {len(names)} 个字段，仅展示前 {len(shown)} 个）"
    return "、".join(shown) + suffix


@dataclass
class OpenAICompatCoach:
    """LLM coach speaking to an OpenAI-compatible chat endpoint.

    Config (API key / base URL / model) comes from `Settings` via the
    underlying client, same as `derive explain`.
    """

    client: LLMClient = field(default_factory=OpenAICompatClient)

    # --- LLMCoach Protocol ---

    def seeds(
        self,
        *,
        label: str,
        industry: str | None,
        available_field_names: list[str],
        available_op_names: list[str],
        k: int = 10,
    ) -> list[ExprNode]:
        user = f"""【任务】基于业务先验，提出最多 {k} 个候选预警表达式（S-表达式）。

【预测目标】{label}
【行业】{industry or "全行业"}
【可用算子】{"、".join(available_op_names)}
【可用字段】{_fields_block(available_field_names)}

输出 JSON：{{"exprs": ["(GT (PctChange 字段名 4) 0.3)", ...]}}"""
        return self._ask_for_trees(user, set(available_op_names), role="seeds")

    def critique(self, *, candidates: list[ExprNode]) -> list[float]:
        neutral = [0.5] * len(candidates)
        if not candidates:
            return []
        listing = "\n".join(
            f"{i}. {c.to_sexpr()}" for i, c in enumerate(candidates)
        )
        user = f"""【任务】对下列候选预警表达式逐一打"业务合理性"分（0~1，0.5 表示中性）。
评分依据：字段组合是否有对公贷后业务含义、方向是否说得通、是否疑似数据噪声拟合。

{listing}

输出 JSON：{{"scores": [0.7, 0.5, ...]}}（长度必须为 {len(candidates)}，顺序对应）"""
        try:
            raw = self.client.complete(system=_SYSTEM, user=user)
            scores = parse_json_response(raw).get("scores")
            if not isinstance(scores, list) or len(scores) != len(candidates):
                logger.warning(
                    "coach.critique returned %s scores for %d candidates; using neutral",
                    None if not isinstance(scores, list) else len(scores),
                    len(candidates),
                )
                return neutral
            return [min(max(float(s), 0.0), 1.0) for s in scores]
        except Exception:
            logger.warning("coach.critique failed; using neutral scores", exc_info=True)
            return neutral

    def cross_source_proposals(
        self,
        *,
        coverage_summary: dict[str, int],
        available_field_names: list[str],
        available_op_names: list[str],
        k: int = 5,
    ) -> list[ExprNode]:
        coverage = "、".join(
            f"{src}={n}" for src, n in sorted(coverage_summary.items(), key=lambda kv: kv[1])
        ) or "（空）"
        user = f"""【任务】当前 Pareto 前沿各数据源的字段引用次数如下（越小越欠探索）：
{coverage}

请提出最多 {k} 个**跨源比较**类候选表达式（S-表达式），优先组合欠探索数据源的字段，
例如"财务报表披露 vs 资金流实际"这类口径不一致信号。

【可用算子】{"、".join(available_op_names)}
【可用字段】{_fields_block(available_field_names)}

输出 JSON：{{"exprs": ["(GT (Div 字段A 字段B) 1.5)", ...]}}"""
        return self._ask_for_trees(user, set(available_op_names), role="cross_source")

    # --- internals ---

    def _ask_for_trees(self, user: str, op_names: set[str], *, role: str) -> list[ExprNode]:
        try:
            raw = self.client.complete(system=_SYSTEM, user=user)
            exprs = parse_json_response(raw).get("exprs")
        except Exception:
            logger.warning("coach.%s failed; returning no trees", role, exc_info=True)
            return []
        if not isinstance(exprs, list):
            logger.warning("coach.%s returned no 'exprs' list; ignoring", role)
            return []
        out: list[ExprNode] = []
        for s in exprs:
            if not isinstance(s, str):
                continue
            try:
                out.append(parse_sexpr(s, op_names))
            except (ValueError, KeyError):
                logger.info("coach.%s proposal failed to parse, skipped: %s", role, s)
        return out
