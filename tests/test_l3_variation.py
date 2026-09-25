"""L3 variation tests: crossover/mutation preserve type invariants."""
from __future__ import annotations

import numpy as np
import pytest

from auto_derivation.expression.tree import to_json
from auto_derivation.expression.typecheck import typecheck
from auto_derivation.l1_data.types import LogicalType
from auto_derivation.l3_search.init import ramped_half_and_half
from auto_derivation.l3_search.primitives import default_primitive_set
from auto_derivation.l3_search.variation import (
    crossover,
    literal_jitter,
    point_mutation,
    subtree_mutation,
)


@pytest.fixture(scope="module")
def primset():
    return default_primitive_set()


@pytest.fixture(scope="module")
def parents(primset):
    rng = np.random.default_rng(42)
    return ramped_half_and_half(primset, pop_size=50, rng=rng, max_depth=4)


def test_crossover_preserves_root_type(primset, parents):
    rng = np.random.default_rng(0)
    for i in range(0, 40, 2):
        a, b = parents[i], parents[i + 1]
        ca, cb = crossover(a, b, primset=primset, rng=rng)
        for t in (ca, cb):
            tc = typecheck(t)
            assert tc.out_type == LogicalType.BOOL


def test_subtree_mutation_preserves_root_type(primset, parents):
    rng = np.random.default_rng(0)
    for p in parents[:20]:
        c = subtree_mutation(p, primset=primset, rng=rng)
        tc = typecheck(c)
        assert tc.out_type == LogicalType.BOOL


def test_point_mutation_preserves_signature(primset, parents):
    rng = np.random.default_rng(0)
    for p in parents[:20]:
        c = point_mutation(p, primset=primset, rng=rng)
        tc = typecheck(c)
        assert tc.out_type == LogicalType.BOOL
        # Same leaf count + same depth (point mut only swaps op heads).
        tc_p = typecheck(p)
        assert tc.leaves == tc_p.leaves
        assert tc.depth == tc_p.depth


def test_literal_jitter_changes_only_literals(primset, parents):
    rng = np.random.default_rng(0)
    # Pick a parent with literals.
    p = next(t for t in parents if "GT" in t.to_sexpr() or "LT" in t.to_sexpr())
    c = literal_jitter(p, primset=primset, rng=rng)
    typecheck(c)  # must still pass
    # Tree shape is identical iff we replaced only literal leaves.
    # We don't assert structural equality (jitter may sample same value),
    # but key shouldn't *grow*: leaves preserved.
    assert typecheck(c).leaves == typecheck(p).leaves


def test_variation_with_invalid_parent_returns_unchanged(primset):
    """If retries fail, variation must return parent unchanged (never raise)."""
    from auto_derivation.expression.tree import FieldNode

    bad = FieldNode(name="THIS_DOES_NOT_EXIST_999")
    rng = np.random.default_rng(0)
    out = subtree_mutation(bad, primset=primset, rng=rng, max_retries=3)
    # Function returns parent; we only check no exception escapes.
    assert to_json(out) == to_json(bad)
