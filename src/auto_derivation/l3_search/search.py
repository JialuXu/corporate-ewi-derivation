"""GP search loop.

Per call this drives one `(label, industry)` slice. DESIGN.md §5.1
mandates partitioning by industry × label — the runner orchestrates that.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from auto_derivation.expression.evaluator import collect_field_names
from auto_derivation.expression.tree import ExprNode, to_canonical_json
from auto_derivation.expression.typecheck import TypeCheckError, typecheck
from auto_derivation.l1_data.panel import join_labels, wide_panel
from auto_derivation.l1_data.registry import default_registry
from auto_derivation.l2_operators.registry import default_operator_registry

from .fitness import INVALID_FITNESS, FitnessConfig, FitnessEvaluator
from .individual import Individual
from .init import ramped_half_and_half
from .llm_coach import LLMCoach, MockLLMCoach
from .nsga2 import assign_ranks, tournament_select
from .primitives import TypedPrimitiveSet
from .variation import crossover, literal_jitter, point_mutation, subtree_mutation

logger = logging.getLogger(__name__)


@dataclass
class GPConfig:
    pop_size: int = 60
    n_gens: int = 20
    init_min_depth: int = 2
    init_max_depth: int = 4
    p_crossover: float = 0.6
    p_subtree_mut: float = 0.2
    p_point_mut: float = 0.1
    p_literal_jitter: float = 0.1
    tournament_size: int = 3
    elitism: int = 4  # carry top-K from previous gen straight through
    seed: int = 0
    # LLM-coach integration (no-ops under MockLLMCoach):
    coach_alpha: float = 0.1
    # ↑ critique soft hint: IV ×= (1 + α·(score − 0.5)), so a neutral 0.5
    #   leaves fitness untouched. Applied once per individual (when fresh).
    coach_call_every: int = 5
    # ↑ ask the coach for cross-source proposals every N generations
    #   (0 disables). Proposals are typechecked and injected after elites.


@dataclass
class SearchResult:
    pareto_front: list[Individual]
    final_population: list[Individual]
    history: list[dict] = field(default_factory=list)


def _make_individual(expr: ExprNode, label: str, industry: str | None) -> Individual:
    return Individual(expr=expr, label=label, industry=industry)


def _vary(
    parents: list[Individual],
    primset: TypedPrimitiveSet,
    rng: np.random.Generator,
    cfg: GPConfig,
) -> list[Individual]:
    out: list[Individual] = []
    i = 0
    while len(out) < len(parents):
        a = parents[i % len(parents)]
        b = parents[(i + 1) % len(parents)]
        i += 1
        u = rng.random()
        if u < cfg.p_crossover:
            ca, cb = crossover(a.expr, b.expr, primset=primset, rng=rng)
            out.append(_make_individual(ca, a.label, a.industry))
            if len(out) < len(parents):
                out.append(_make_individual(cb, a.label, a.industry))
        elif u < cfg.p_crossover + cfg.p_subtree_mut:
            child = subtree_mutation(a.expr, primset=primset, rng=rng)
            out.append(_make_individual(child, a.label, a.industry))
        elif u < cfg.p_crossover + cfg.p_subtree_mut + cfg.p_point_mut:
            child = point_mutation(a.expr, primset=primset, rng=rng)
            out.append(_make_individual(child, a.label, a.industry))
        else:
            child = literal_jitter(a.expr, primset=primset, rng=rng)
            out.append(_make_individual(child, a.label, a.industry))
    return out


def _available_field_names(primset: TypedPrimitiveSet) -> list[str]:
    return [
        f.name_cn or f.metric_id
        for fs in primset._fields_by_type.values()
        for f in fs
    ]


def _typechecked_individuals(
    trees: list[ExprNode], label: str, industry: str | None
) -> list[Individual]:
    out: list[Individual] = []
    for tree in trees:
        try:
            typecheck(tree)
        except (TypeCheckError, KeyError):
            continue
        out.append(_make_individual(tree, label, industry))
    return out


def _seed_individuals(
    coach: LLMCoach,
    label: str,
    industry: str | None,
    primset: TypedPrimitiveSet,
    k: int,
) -> list[Individual]:
    raw = coach.seeds(
        label=label,
        industry=industry,
        available_field_names=_available_field_names(primset),
        available_op_names=primset.op_registry.names(),
        k=k,
    )
    return _typechecked_individuals(raw, label, industry)


def _apply_critique(
    coach: LLMCoach, fresh: list[Individual], alpha: float
) -> None:
    """Soft business-plausibility hint on newly evaluated individuals:
    IV ×= (1 + α·(score − 0.5)). Centered so a neutral 0.5 changes nothing,
    and applied only when the individual is fresh so elites carried across
    generations are not compounded."""
    targets = [
        ind for ind in fresh
        if ind.fitness is not None and ind.fitness != INVALID_FITNESS
    ]
    if not targets or alpha <= 0:
        return
    scores = coach.critique(candidates=[t.expr for t in targets])
    if len(scores) != len(targets):
        logger.warning(
            "coach.critique returned %d scores for %d candidates; skipping hint",
            len(scores), len(targets),
        )
        return
    for ind, s in zip(targets, scores, strict=True):
        f = ind.fitness
        assert f is not None  # narrowed by the filter above
        s = min(max(float(s), 0.0), 1.0)
        ind.fitness = (f[0] * (1.0 + alpha * (s - 0.5)), *f[1:])


def _cross_source_individuals(
    coach: LLMCoach,
    pop: list[Individual],
    front0_idx: list[int],
    primset: TypedPrimitiveSet,
    label: str,
    industry: str | None,
    k: int = 5,
) -> list[Individual]:
    """Ask the coach for under-explored cross-source combinations, using the
    current front-0 field references as the coverage summary."""
    coverage: dict[str, int] = {}
    for i in front0_idx:
        for fname in collect_field_names(pop[i].expr):
            try:
                mid = primset.metric_registry.resolve(fname).metric_id
            except KeyError:
                continue
            src = mid.split("_", 1)[0]
            coverage[src] = coverage.get(src, 0) + 1
    raw = coach.cross_source_proposals(
        coverage_summary=coverage,
        available_field_names=_available_field_names(primset),
        available_op_names=primset.op_registry.names(),
        k=k,
    )
    return _typechecked_individuals(raw, label, industry)


def search(
    panel_path: Path,
    labels_path: Path,
    *,
    label: str,
    industry: str | None = None,
    config: GPConfig | None = None,
    coach: LLMCoach | None = None,
    fitness_cfg: FitnessConfig | None = None,
) -> SearchResult:
    """Run GP search on one (label, industry) slice. Returns Pareto front +
    final population + history."""
    cfg = config or GPConfig()
    coach = coach or MockLLMCoach()
    rng = np.random.default_rng(cfg.seed)

    # Build the working panel: wide pivot + label join (+ optional industry filter).
    panel = wide_panel(panel_path)
    panel = join_labels(panel, labels_path, label_name=label)
    if industry:
        panel = panel.filter(pl.col("industry") == industry)

    available_metric_ids = frozenset(
        c for c in panel.columns
        if c not in ("entity_id", "observation_date", "industry", label)
    )
    primset = TypedPrimitiveSet(
        op_registry=default_operator_registry(),
        metric_registry=default_registry(),
        field_whitelist=available_metric_ids,
    )

    fc = fitness_cfg or FitnessConfig(label_col=label, industry=industry)
    # Make sure the evaluator's industry filter agrees with the (already-filtered) panel.
    fc.industry = None if industry else fc.industry
    evaluator = FitnessEvaluator(panel=panel, config=fc)

    # 1) Initial population: LLM seeds + random.
    seeds = _seed_individuals(coach, label, industry, primset, k=min(10, cfg.pop_size // 2))
    n_random = cfg.pop_size - len(seeds)
    random_trees = ramped_half_and_half(
        primset,
        pop_size=n_random,
        min_depth=cfg.init_min_depth,
        max_depth=cfg.init_max_depth,
        rng=rng,
    )
    pop: list[Individual] = seeds + [_make_individual(t, label, industry) for t in random_trees]

    history: list[dict] = []

    for gen in range(cfg.n_gens):
        # Evaluate (fills in `fitness`).
        fresh = [ind for ind in pop if ind.fitness is None]
        for ind in fresh:
            ind.fitness = evaluator.evaluate(ind.expr)
        # Coach role 2 (pruning advisor): soft plausibility hint on this
        # generation's newly evaluated individuals.
        _apply_critique(coach, fresh, cfg.coach_alpha)

        # Rank + crowding.
        fronts = assign_ranks(pop)

        # Bookkeeping.
        valid_fits: list[tuple[float, ...]] = [
            p.fitness for p in pop
            if p.fitness is not None and p.fitness != INVALID_FITNESS
        ]
        best_iv = max((f[0] for f in valid_fits), default=float("nan"))
        best_ks = max((f[1] for f in valid_fits), default=float("nan"))
        history.append(
            {
                "gen": gen,
                "valid": len(valid_fits),
                "n_fronts": len(fronts),
                "front0_size": len(fronts[0]) if fronts else 0,
                "best_iv": best_iv,
                "best_ks": best_ks,
            }
        )
        logger.info(
            "gen=%2d valid=%3d front0=%3d best_iv=%.3f best_ks=%.3f",
            gen, len(valid_fits), len(fronts[0]) if fronts else 0, best_iv, best_ks,
        )

        if gen == cfg.n_gens - 1:
            break

        # Selection — elitism carries top-`cfg.elitism` from front 0.
        pop_sorted = sorted(pop, key=lambda p: (p.rank, -p.crowding))
        elites = pop_sorted[: cfg.elitism]
        # Deep-copy elites (preserve fitness so we don't re-evaluate next gen).
        new_pop: list[Individual] = [
            Individual(
                expr=e.expr, label=e.label, industry=e.industry,
                fitness=e.fitness, rank=e.rank, crowding=e.crowding,
            )
            for e in elites
        ]
        # Coach role 3 (cross-source proposer): every N generations inject
        # typechecked proposals right after the elites.
        if cfg.coach_call_every > 0 and (gen + 1) % cfg.coach_call_every == 0:
            proposals = _cross_source_individuals(
                coach, pop, fronts[0] if fronts else [], primset, label, industry,
            )
            existing = {to_canonical_json(e.expr) for e in new_pop}
            for ind in proposals:
                key = to_canonical_json(ind.expr)
                if key in existing or len(new_pop) >= cfg.pop_size:
                    continue
                existing.add(key)
                new_pop.append(ind)
        parents = tournament_select(
            pop, k=cfg.pop_size - len(new_pop), rng=rng,
            tournament_size=cfg.tournament_size,
        )
        offspring = _vary(parents, primset, rng, cfg)
        # Drop duplicates (keep elites + first occurrence of each new tree).
        seen_keys = {to_canonical_json(e.expr) for e in new_pop}
        for child in offspring:
            k = to_canonical_json(child.expr)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            new_pop.append(child)
        # If too few unique offspring, top up with fresh random trees.
        while len(new_pop) < cfg.pop_size:
            extras = ramped_half_and_half(
                primset, pop_size=1, min_depth=cfg.init_min_depth,
                max_depth=cfg.init_max_depth, rng=rng,
            )
            for t in extras:
                k = to_canonical_json(t)
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                new_pop.append(_make_individual(t, label, industry))
        pop = new_pop[: cfg.pop_size]

    # Final ranking.
    for ind in pop:
        if ind.fitness is None:
            ind.fitness = evaluator.evaluate(ind.expr)
    final_fronts = assign_ranks(pop)
    pareto_front = [pop[i] for i in final_fronts[0]] if final_fronts else []
    pareto_front = [p for p in pareto_front if p.fitness != INVALID_FITNESS]

    return SearchResult(
        pareto_front=pareto_front,
        final_population=pop,
        history=history,
    )
