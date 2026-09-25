"""Tests for `persistence` — P1.1 search/explanation artifact layout.

These check the file layout, JSON shape, and idempotency contract that
downstream review tooling will rely on. We don't run a real GP search here —
we hand-roll a couple of `Individual`s with synthetic fitness vectors.
"""
from __future__ import annotations

import json

from auto_derivation.expression.tree import (
    FieldNode,
    LiteralNode,
    OpNode,
    parse_sexpr,
)
from auto_derivation.l3_search.fitness import FitnessConfig
from auto_derivation.l3_search.individual import Individual
from auto_derivation.l3_search.search import GPConfig
from auto_derivation.l4_explain.card import ExplanationCard, ExplanationResult
from auto_derivation.persistence import (
    _expr_sha12,
    save_explanation_card,
    save_search_run,
)


def _toy_individual(
    expr_sexpr: str = "(GT (PctChange 营业收入 1) -0.3)",
    fitness=(0.4, 0.3, 0.8, 0.6, -3.0, -0.1),
) -> Individual:
    tree = parse_sexpr(expr_sexpr, op_names={"GT", "PctChange"})
    return Individual(
        expr=tree, label="Y_npl_12m", industry=None,
        fitness=fitness, rank=0, crowding=1.5,
    )


# ---------- save_search_run ----------


def test_save_search_run_writes_three_files(tmp_path):
    inds = [_toy_individual(), _toy_individual(fitness=(0.2, 0.1, 0.7, 0.5, -4.0, -0.2))]
    history = [{"gen": 0, "valid": 10, "n_fronts": 2, "front0_size": 4,
                "best_iv": 0.4, "best_ks": 0.3}]
    cfg = GPConfig(pop_size=20, n_gens=2, seed=7)
    fc = FitnessConfig(label_col="Y_npl_12m", industry=None)

    run_dir = save_search_run(
        pareto=inds, history=history,
        label="Y_npl_12m", scope=None,
        gp_config=cfg, fitness_config=fc,
        out_dir=tmp_path,
    )
    assert run_dir.parent == tmp_path
    assert (run_dir / "pareto.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "config.json").exists()


def test_save_search_run_pareto_payload_shape(tmp_path):
    inds = [_toy_individual()]
    run_dir = save_search_run(
        pareto=inds, history=[],
        label="Y_npl_12m", scope=None,
        gp_config=GPConfig(), fitness_config=FitnessConfig(label_col="Y_npl_12m"),
        out_dir=tmp_path,
    )
    payload = json.loads((run_dir / "pareto.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    row = payload[0]
    assert set(row) == {"expr_dict", "expr_sexpr", "fitness", "rank", "crowding"}
    # round-trip: expr_dict reconstructs the same tree
    assert row["expr_dict"]["kind"] == "op"
    assert row["expr_dict"]["op"] == "GT"
    assert row["fitness"] == [0.4, 0.3, 0.8, 0.6, -3.0, -0.1]


def test_save_search_run_appends_index_jsonl(tmp_path):
    inds = [_toy_individual()]
    save_search_run(
        pareto=inds, history=[],
        label="Y_npl_12m", scope="制造",
        gp_config=GPConfig(), fitness_config=FitnessConfig(label_col="Y_npl_12m"),
        out_dir=tmp_path,
    )
    save_search_run(
        pareto=inds, history=[],
        label="Y_overdue_3m", scope=None,
        gp_config=GPConfig(), fitness_config=FitnessConfig(label_col="Y_overdue_3m"),
        out_dir=tmp_path,
    )
    index = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index) == 2
    rows = [json.loads(line) for line in index]
    assert rows[0]["label"] == "Y_npl_12m" and rows[0]["scope"] == "制造"
    assert rows[1]["label"] == "Y_overdue_3m" and rows[1]["scope"] is None
    for r in rows:
        assert r["best_iv"] == 0.4
        assert r["n_pareto"] == 1


def test_save_search_run_config_records_lineage(tmp_path):
    inds = [_toy_individual()]
    cfg = GPConfig(pop_size=42, seed=7)
    fc = FitnessConfig(label_col="Y_npl_12m", train_frac=0.5, valid_frac=0.25)
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"not really parquet, just bytes for sha")

    run_dir = save_search_run(
        pareto=inds, history=[],
        label="Y_npl_12m", scope=None,
        gp_config=cfg, fitness_config=fc,
        panel_path=panel_path, out_dir=tmp_path,
    )
    cfg_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_payload["label"] == "Y_npl_12m"
    assert cfg_payload["gp_config"]["pop_size"] == 42
    assert cfg_payload["gp_config"]["seed"] == 7
    assert cfg_payload["fitness_config"]["train_frac"] == 0.5
    assert cfg_payload["panel_sha256"] is not None
    assert len(cfg_payload["panel_sha256"]) == 64
    # git_sha may be None in some environments — just check the key exists.
    assert "git_sha" in cfg_payload


def test_save_search_run_handles_invalid_fitness_in_index(tmp_path):
    """A pareto member with fitness=None must not crash the index summary."""
    ind = _toy_individual()
    ind.fitness = None
    save_search_run(
        pareto=[ind], history=[],
        label="Y_npl_12m", scope=None,
        gp_config=GPConfig(), fitness_config=FitnessConfig(label_col="Y_npl_12m"),
        out_dir=tmp_path,
    )
    row = json.loads((tmp_path / "index.jsonl").read_text(encoding="utf-8").strip())
    assert row["best_iv"] is None
    assert row["best_ks"] is None
    assert row["n_pareto"] == 1


# ---------- save_explanation_card ----------


def _toy_card() -> ExplanationCard:
    return ExplanationCard(
        metric_name_cn="测试指标",
        business_explanation="解释",
        use_cases="适用",
        diff_vs_existing="差异",
        threshold_advice="阈值",
        review_checklist=["核对1", "核对2"],
    )


def test_save_explanation_card_layout(tmp_path):
    expr = parse_sexpr("(GT (PctChange 营业收入 1) -0.3)", op_names={"GT", "PctChange"})
    result = ExplanationResult(card=_toy_card(), consistency_warnings=["⚠ 阈值方向"])

    path = save_explanation_card(expr=expr, label="Y_npl_12m", result=result, out_dir=tmp_path)
    sha12 = _expr_sha12(expr)
    assert path == tmp_path / sha12[:2] / f"{sha12}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["expr_sha12"] == sha12
    assert payload["card"]["metric_name_cn"] == "测试指标"
    assert payload["consistency_warnings"] == ["⚠ 阈值方向"]
    assert payload["label"] == "Y_npl_12m"


def test_save_explanation_card_is_idempotent_on_same_expr(tmp_path):
    expr = parse_sexpr("(GT (PctChange 营业收入 1) -0.3)", op_names={"GT", "PctChange"})
    r1 = ExplanationResult(card=_toy_card(), consistency_warnings=[])
    p1 = save_explanation_card(expr=expr, label="Y_npl_12m", result=r1, out_dir=tmp_path)

    new_card = _toy_card()
    new_card_dict = new_card.model_dump()
    new_card_dict["business_explanation"] = "更新后的解释"
    r2 = ExplanationResult(card=ExplanationCard.model_validate(new_card_dict),
                           consistency_warnings=[])
    p2 = save_explanation_card(expr=expr, label="Y_npl_12m", result=r2, out_dir=tmp_path)
    assert p1 == p2
    payload = json.loads(p2.read_text(encoding="utf-8"))
    assert payload["card"]["business_explanation"] == "更新后的解释"


def test_expr_sha12_stable_across_construction_styles():
    """sha12 is content-addressed — manual node construction must equal the
    parsed S-expression."""
    parsed = parse_sexpr("(GT (PctChange 营业收入 1) -0.3)", op_names={"GT", "PctChange"})
    pct_change = OpNode(
        op_name="PctChange",
        children=(FieldNode(name="营业收入"), LiteralNode(value=1)),
    )
    manual = OpNode(
        op_name="GT",
        children=(pct_change, LiteralNode(value=-0.3)),
    )
    assert _expr_sha12(parsed) == _expr_sha12(manual)
