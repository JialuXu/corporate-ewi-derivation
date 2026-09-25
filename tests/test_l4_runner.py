"""End-to-end L4 runner test using MockClient."""
from __future__ import annotations

import json

from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.l2_operators.registry import default_operator_registry
from auto_derivation.l4_explain.llm_client import MockClient, parse_json_response
from auto_derivation.l4_explain.runner import explain


def test_runner_with_mock_client(synthetic_paths):
    panel_path, labels_path = synthetic_paths
    op_names = set(default_operator_registry().names())
    expr = parse_sexpr("(GT (PctChange 营业收入 12) -0.1)", op_names)

    canned = json.dumps(
        {
            "metric_name_cn": "营收同比下滑预警",
            "business_explanation": "当 营业收入 同比变化高于阈值时触发，提示经营弱化。",
            "use_cases": "制造业、批零业对公客户",
            "diff_vs_existing": "对短周期下滑更敏感",
            "threshold_advice": "PctChange < -0.1 时强预警",
            "review_checklist": ["核查营收口径", "比对资金流"],
        },
        ensure_ascii=False,
    )
    client = MockClient(canned_json=canned)
    result = explain(
        expr,
        panel_path=panel_path,
        labels_path=labels_path,
        label="Y_npl_12m",
        llm_client=client,
    )

    assert result.card.metric_name_cn == "营收同比下滑预警"
    # Mock call captured both system + user prompts.
    assert client.last_call is not None
    assert "S-表达式" in client.last_call["user"]
    # Consistency: no warnings since 营业收入 is in the tree and direction matches.
    assert result.consistency_warnings == []


def test_runner_with_prepivoted_panel(synthetic_paths):
    """Batch explainers pivot once and pass `panel=` — no panel_path needed."""
    from auto_derivation.l1_data.panel import wide_panel

    panel_path, labels_path = synthetic_paths
    op_names = set(default_operator_registry().names())
    expr = parse_sexpr("(GT (PctChange 营业收入 12) -0.1)", op_names)

    canned = json.dumps(
        {
            "metric_name_cn": "营收同比下滑预警",
            "business_explanation": "当 营业收入 同比变化高于阈值时触发。",
            "use_cases": "对公客户",
            "diff_vs_existing": "无",
            "threshold_advice": "PctChange < -0.1 时强预警",
            "review_checklist": ["核查营收口径"],
            "referenced_fields": ["营业收入"],
        },
        ensure_ascii=False,
    )
    panel = wide_panel(panel_path)
    result = explain(
        expr,
        labels_path=labels_path,
        label="Y_npl_12m",
        llm_client=MockClient(canned_json=canned),
        panel=panel,
    )
    assert result.card.referenced_fields == ["营业收入"]
    assert result.consistency_warnings == []


def test_parse_json_strips_code_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert parse_json_response(raw) == {"a": 1}


def test_parse_json_finds_object_in_prose():
    raw = "这是回答：{\"a\": 2}\n谢谢"
    assert parse_json_response(raw) == {"a": 2}
