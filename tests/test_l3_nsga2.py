"""NSGA-II ranking + crowding tests."""
from __future__ import annotations

import math

from auto_derivation.expression.tree import FieldNode
from auto_derivation.l3_search.individual import Individual
from auto_derivation.l3_search.nsga2 import (
    assign_ranks,
    crowding_distance,
    dominates,
    fast_nondominated_sort,
)


def _ind(fitness):
    return Individual(expr=FieldNode(name="x"), label="Y", fitness=fitness)


def test_dominates_basic():
    assert dominates((1.0, 1.0), (0.0, 0.0))
    assert dominates((1.0, 0.5), (0.5, 0.5))
    assert not dominates((1.0, 0.0), (0.0, 1.0))  # trade-off
    assert not dominates((1.0, 1.0), (1.0, 1.0))  # equal


def test_fast_nondominated_sort():
    # Pareto front: (0.9,0.1) (0.5,0.5) (0.1,0.9); (0.4,0.4) dominated by (0.5,0.5).
    pop = [_ind(f) for f in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.4, 0.4), (0.0, 0.0)]]
    fronts = fast_nondominated_sort(pop)
    assert len(fronts[0]) == 3
    assert {pop[i].fitness for i in fronts[0]} == {(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)}
    assert {pop[i].fitness for i in fronts[1]} == {(0.4, 0.4)}
    assert {pop[i].fitness for i in fronts[2]} == {(0.0, 0.0)}


def test_crowding_distance_endpoints_infinite():
    pop = [_ind(f) for f in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]]
    d = crowding_distance(pop, [0, 1, 2])
    # Endpoints (extremes per objective) get inf; middle gets finite.
    inf_count = sum(1 for v in d.values() if math.isinf(v))
    assert inf_count == 2
    finite = [v for v in d.values() if not math.isinf(v)]
    assert finite[0] > 0


def test_assign_ranks_sets_all_individuals():
    pop = [_ind((0.0, 0.0)), _ind((1.0, 1.0)), _ind(None)]  # one invalid
    assign_ranks(pop)
    assert pop[1].rank == 0
    assert pop[0].rank == 1
    assert pop[2].rank > 1000  # sentinel for invalid
