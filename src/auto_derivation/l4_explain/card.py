"""ExplanationCard — pydantic model matching the card template in DESIGN.md §6.1."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExplanationCard(BaseModel):
    """Business-facing card describing one auto-derived warning rule."""

    metric_name_cn: str = Field(
        ...,
        description="中文指标名，≤15字，体现业务含义",
    )
    business_explanation: str = Field(
        ..., description="2-3句话讲明触发逻辑与风险含义"
    )
    use_cases: str = Field(..., description="适用客户类型 / 时点")
    diff_vs_existing: str = Field(
        ..., description="与已有指标的差异点"
    )
    threshold_advice: str = Field(..., description="建议的预警阈值范围")
    review_checklist: list[str] = Field(
        ...,
        description="业务复核时应额外查的事项",
        min_length=1,
        max_length=10,
    )
    referenced_fields: list[str] = Field(
        default_factory=list,
        description="解释中实际引用的原子指标中文名列表（结构化字段核对通道；"
        "为空时一致性检查回退到对全文的子串扫描）",
    )

    @field_validator("metric_name_cn")
    @classmethod
    def _name_short_enough(cls, v: str) -> str:
        if len(v) > 15:
            raise ValueError(
                f"metric_name_cn must be ≤15 chars (got {len(v)}: {v!r})"
            )
        return v


class ExplanationResult(BaseModel):
    card: ExplanationCard
    consistency_warnings: list[str] = Field(default_factory=list)
    raw_llm_output: str | None = None
