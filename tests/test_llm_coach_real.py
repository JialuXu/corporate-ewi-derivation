"""OpenAICompatCoach tests — all via MockClient, no API calls.

The coach must never raise into the GP loop: bad JSON, wrong-length score
lists and unparsable S-expressions all degrade to neutral defaults.
"""
from __future__ import annotations

import json

from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.l2_operators.registry import default_operator_registry
from auto_derivation.l3_search.llm_coach_real import OpenAICompatCoach
from auto_derivation.l4_explain.llm_client import MockClient

OP_NAMES = list(default_operator_registry().names())
FIELDS = ["营业收入", "近一年被执行案件数量"]


def _coach(canned: str) -> OpenAICompatCoach:
    return OpenAICompatCoach(client=MockClient(canned_json=canned))


def test_seeds_parses_valid_and_skips_invalid():
    canned = json.dumps(
        {"exprs": ["(GT (PctChange 营业收入 12) -0.1)", "(NoSuchOp 营业收入)", 42]},
        ensure_ascii=False,
    )
    trees = _coach(canned).seeds(
        label="Y_npl_12m", industry=None,
        available_field_names=FIELDS, available_op_names=OP_NAMES,
    )
    assert len(trees) == 1
    expected = parse_sexpr("(GT (PctChange 营业收入 12) -0.1)", set(OP_NAMES))
    assert trees[0].to_sexpr() == expected.to_sexpr()


def test_seeds_bad_json_returns_empty():
    trees = _coach("not json at all").seeds(
        label="Y_npl_12m", industry=None,
        available_field_names=FIELDS, available_op_names=OP_NAMES,
    )
    assert trees == []


def test_critique_clamps_scores():
    coach = _coach(json.dumps({"scores": [1.7, -0.2]}))
    candidates = [
        parse_sexpr("(GT 营业收入 0.0)", set(OP_NAMES)),
        parse_sexpr("(LT 营业收入 0.0)", set(OP_NAMES)),
    ]
    assert coach.critique(candidates=candidates) == [1.0, 0.0]


def test_critique_wrong_length_returns_neutral():
    coach = _coach(json.dumps({"scores": [0.9]}))
    candidates = [
        parse_sexpr("(GT 营业收入 0.0)", set(OP_NAMES)),
        parse_sexpr("(LT 营业收入 0.0)", set(OP_NAMES)),
    ]
    assert coach.critique(candidates=candidates) == [0.5, 0.5]


def test_critique_empty_candidates_no_llm_call():
    client = MockClient(canned_json="{}")
    coach = OpenAICompatCoach(client=client)
    assert coach.critique(candidates=[]) == []
    assert client.last_call is None


def test_cross_source_proposals_parse():
    canned = json.dumps({"exprs": ["(GT 营业收入 0.0)"]}, ensure_ascii=False)
    trees = _coach(canned).cross_source_proposals(
        coverage_summary={"FIN": 3, "JUD": 0},
        available_field_names=FIELDS, available_op_names=OP_NAMES,
    )
    assert len(trees) == 1
