"""NSGA-II — fast non-dominated sort + crowding distance.

We treat every fitness component as "maximize" (callers sign-flip things they
want minimized — see fitness.py: complexity_neg, redundancy_neg).

Implementation follows Deb 2002. Time complexity: O(M·N²) for fronts +
O(M·N·log N) for crowding distances.
"""
from __future__ import annotations

import math

import numpy as np

from .individual import Individual

INVALID_RANK = 1_000_000  # individuals with no valid fitness sink to the bottom


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Pareto domination: a dominates b iff a ≥ b in every objective and
    strictly better in at least one."""
    better_in_one = False
    for ai, bi in zip(a, b, strict=True):
        if ai < bi:
            return False
        if ai > bi:
            better_in_one = True
    return better_in_one


def fast_nondominated_sort(pop: list[Individual]) -> list[list[int]]:
    """Return a list of fronts; each front is a list of indices into `pop`.
    Front 0 is the Pareto-optimal set."""
    n = len(pop)
    s: list[list[int]] = [[] for _ in range(n)]  # who I dominate
    n_dom = [0] * n  # how many dominate me
    fronts: list[list[int]] = [[]]

    valid_idx = [i for i, ind in enumerate(pop) if ind.fitness is not None]
    for p in valid_idx:
        for q in valid_idx:
            if p == q:
                continue
            fp = pop[p].fitness or ()
            fq = pop[q].fitness or ()
            if dominates(fp, fq):
                s[p].append(q)
            elif dominates(fq, fp):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in s[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def crowding_distance(pop: list[Individual], front: list[int]) -> dict[int, float]:
    """Compute crowding distance for one front. Returns {pop_index: distance}."""
    if not front:
        return {}
    dist: dict[int, float] = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        for i in front:
            dist[i] = math.inf
        return dist
    fitnesses: list[tuple[float, ...]] = [
        pop[i].fitness for i in front if pop[i].fitness is not None  # type: ignore[misc]
    ]
    if not fitnesses:
        return dist
    n_obj = len(fitnesses[0])
    for m in range(n_obj):
        sorted_idx = sorted(front, key=lambda i: (pop[i].fitness or (0,) * n_obj)[m])
        f_min = (pop[sorted_idx[0]].fitness or (0,) * n_obj)[m]
        f_max = (pop[sorted_idx[-1]].fitness or (0,) * n_obj)[m]
        dist[sorted_idx[0]] = math.inf
        dist[sorted_idx[-1]] = math.inf
        if f_max - f_min == 0:
            continue
        for k in range(1, len(sorted_idx) - 1):
            prev_v = (pop[sorted_idx[k - 1]].fitness or (0,) * n_obj)[m]
            next_v = (pop[sorted_idx[k + 1]].fitness or (0,) * n_obj)[m]
            dist[sorted_idx[k]] += (next_v - prev_v) / (f_max - f_min)
    return dist


def assign_ranks(pop: list[Individual]) -> list[list[int]]:
    """Run NSGA-II ranking + crowding on `pop` in place. Returns fronts."""
    fronts = fast_nondominated_sort(pop)
    for rank, front in enumerate(fronts):
        d = crowding_distance(pop, front)
        for i in front:
            pop[i].rank = rank
            pop[i].crowding = d[i]
    # Anyone left unranked (no valid fitness) gets a sentinel rank.
    for ind in pop:
        if ind.rank == -1:
            ind.rank = INVALID_RANK
            ind.crowding = 0.0
    return fronts


def tournament_select(
    pop: list[Individual],
    k: int,
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> list[Individual]:
    """Binary-style tournament selection (configurable size). Picks `k`
    individuals; selection criterion is (lower rank, higher crowding)."""
    out: list[Individual] = []
    n = len(pop)
    for _ in range(k):
        idxs = rng.integers(0, n, size=tournament_size)
        best = pop[int(idxs[0])]
        for j in idxs[1:]:
            cand = pop[int(j)]
            if (cand.rank, -cand.crowding) < (best.rank, -best.crowding):
                best = cand
        out.append(best)
    return out
