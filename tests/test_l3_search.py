"""End-to-end GP search test on synthetic data.

Regression: with the injected revenue-decline + judicial signal, the search
should consistently find a Pareto-front member with IV ≥ 0.20 within a
small generation budget.
"""
from __future__ import annotations

from auto_derivation.l3_search.fitness import INVALID_FITNESS
from auto_derivation.l3_search.llm_coach import MockLLMCoach
from auto_derivation.l3_search.search import GPConfig, search


def test_search_finds_signal_on_synthetic(synthetic_paths):
    panel_path, labels_path = synthetic_paths
    cfg = GPConfig(pop_size=40, n_gens=8, seed=0)
    res = search(
        panel_path,
        labels_path,
        label="Y_npl_12m",
        config=cfg,
        coach=MockLLMCoach(),
    )

    assert len(res.history) == cfg.n_gens
    assert len(res.pareto_front) > 0

    valid_front = [p for p in res.pareto_front if p.fitness != INVALID_FITNESS]
    assert valid_front, "Pareto front contained only invalid individuals"

    front_fits = [p.fitness for p in valid_front if p.fitness is not None]
    best_iv = max(f[0] for f in front_fits)
    assert best_iv >= 0.20, (
        f"GP failed to find IV ≥ 0.20 on synthetic data; got {best_iv:.3f}. "
        f"Expressions on Pareto front: "
        f"{[p.expr.to_sexpr() for p in valid_front[:5]]}"
    )

    # Search should make progress: final-gen best_iv > first-gen best_iv.
    first_iv = res.history[0]["best_iv"]
    final_iv = res.history[-1]["best_iv"]
    assert final_iv >= first_iv, "GP regressed across generations"


def test_search_per_industry(synthetic_paths):
    panel_path, labels_path = synthetic_paths
    cfg = GPConfig(pop_size=20, n_gens=3, seed=1)
    res = search(
        panel_path, labels_path,
        label="Y_npl_12m", industry="制造", config=cfg,
    )
    # At least one valid candidate within the small slice.
    assert any(p.fitness != INVALID_FITNESS for p in res.final_population)


def test_fitness_cache_merges_equivalent_literals():
    """LiteralNode(0) and LiteralNode(0.0) must share one cache entry."""
    import polars as pl

    from auto_derivation.expression.tree import FieldNode, LiteralNode, OpNode
    from auto_derivation.l3_search.fitness import FitnessConfig, FitnessEvaluator

    evaluator = FitnessEvaluator(panel=pl.DataFrame(), config=FitnessConfig())
    # Unknown field → typecheck fails → INVALID, but the cache key is still
    # written, which is all this test needs.
    int_lit = OpNode("GT", (FieldNode("不存在的字段"), LiteralNode(0)))
    float_lit = OpNode("GT", (FieldNode("不存在的字段"), LiteralNode(0.0)))
    assert evaluator.evaluate(int_lit) == evaluator.evaluate(float_lit)
    assert len(evaluator._cache) == 1


class _SpyCoach:
    """Counts coach calls and injects one fixed cross-source proposal."""

    def __init__(self, proposal):
        self.proposal = proposal
        self.critique_calls = 0
        self.cross_calls = 0

    def seeds(self, *, label, industry, available_field_names, available_op_names, k=10):
        return []

    def critique(self, *, candidates):
        self.critique_calls += 1
        return [0.5] * len(candidates)

    def cross_source_proposals(
        self, *, coverage_summary, available_field_names, available_op_names, k=5
    ):
        self.cross_calls += 1
        assert isinstance(coverage_summary, dict)
        return [self.proposal]


def test_coach_critique_and_proposals_are_wired(synthetic_paths):
    from auto_derivation.expression.tree import parse_sexpr, to_json
    from auto_derivation.l2_operators.registry import default_operator_registry

    panel_path, labels_path = synthetic_paths
    op_names = set(default_operator_registry().names())
    proposal = parse_sexpr("(GT (PctChange 营业收入 12) -0.1)", op_names)

    coach = _SpyCoach(proposal)
    cfg = GPConfig(pop_size=12, n_gens=4, seed=0, coach_call_every=1)
    res = search(
        panel_path, labels_path,
        label="Y_npl_12m", config=cfg, coach=coach,
    )

    # critique() runs on the fresh individuals of each generation.
    assert coach.critique_calls >= 1
    # cross_source_proposals() runs every coach_call_every gens during
    # selection (the final generation has no selection step).
    assert coach.cross_calls == cfg.n_gens - 1
    # The injected proposal survives into the final population.
    assert any(
        to_json(p.expr) == to_json(proposal) for p in res.final_population
    ), "cross-source proposal was not injected"
