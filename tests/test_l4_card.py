"""ExplanationCard pydantic model tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from auto_derivation.l4_explain.card import ExplanationCard


def _valid_payload(**overrides):
    base = {
        "metric_name_cn": "营收下滑预警",
        "business_explanation": "近期营业收入连续下滑且低于行业均值时触发，提示经营恶化。",
        "use_cases": "制造业 / 批零业对公客户",
        "diff_vs_existing": "比单期同比下滑更敏感，强调持续性。",
        "threshold_advice": "PctChange < -0.2 时强预警",
        "review_checklist": ["核查营收口径", "比对资金流"],
    }
    base.update(overrides)
    return base


def test_valid_card():
    c = ExplanationCard.model_validate(_valid_payload())
    assert c.metric_name_cn == "营收下滑预警"
    assert len(c.review_checklist) == 2


def test_metric_name_too_long_rejected():
    with pytest.raises(ValidationError):
        ExplanationCard.model_validate(
            _valid_payload(metric_name_cn="超过十五个字符的指标名字真的不能这么长")
        )


def test_review_checklist_must_be_nonempty():
    with pytest.raises(ValidationError):
        ExplanationCard.model_validate(_valid_payload(review_checklist=[]))
