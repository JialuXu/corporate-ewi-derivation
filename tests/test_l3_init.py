"""L3 init tests: ramped half-and-half produces valid typed trees."""
from __future__ import annotations

import numpy as np
import pytest

from auto_derivation.expression.tree import to_json
from auto_derivation.expression.typecheck import typecheck
from auto_derivation.l1_data.types import LogicalType
from auto_derivation.l3_search.init import grow_tree, ramped_half_and_half
from auto_derivation.l3_search.primitives import default_primitive_set


@pytest.fixture(scope="module")
def primset():
    return default_primitive_set()


def test_grow_tree_root_is_bool(primset):
    rng = np.random.default_rng(0)
    for _ in range(20):
        try:
            t = grow_tree(primset, rng, min_depth=2, max_depth=4)
        except Exception:
            continue
        tc = typecheck(t)
        assert tc.out_type == LogicalType.BOOL


def test_ramped_half_and_half_unique_and_valid(primset):
    rng = np.random.default_rng(1)
    pop = ramped_half_and_half(primset, pop_size=30, rng=rng, max_depth=4)
    assert len(pop) == 30
    keys = {to_json(t) for t in pop}
    assert len(keys) == 30, "trees should be unique"
    for t in pop:
        tc = typecheck(t)
        assert tc.out_type == LogicalType.BOOL
        assert tc.depth <= 4
        assert tc.leaves <= 8


def test_field_whitelist_respected():
    from auto_derivation.l1_data.registry import default_registry
    from auto_derivation.l2_operators.registry import default_operator_registry
    from auto_derivation.l3_search.primitives import TypedPrimitiveSet

    # Whitelist exactly two numeric metrics — every produced field must be one of them.
    whitelist = frozenset({"FIN_0001", "FIN_0004"})
    ps = TypedPrimitiveSet(
        op_registry=default_operator_registry(),
        metric_registry=default_registry(),
        field_whitelist=whitelist,
    )
    rng = np.random.default_rng(2)
    pop = ramped_half_and_half(ps, pop_size=15, rng=rng, max_depth=3)
    for t in pop:
        s = t.to_sexpr()
        # Field names shown will be the metric_id since name_cn lookup pulls
        # from registry; check that metric_ids referenced are in whitelist.
        from auto_derivation.expression.evaluator import collect_field_names
        from auto_derivation.l1_data.registry import default_registry as dr

        reg = dr()
        for fname in collect_field_names(t):
            assert reg.resolve(fname).metric_id in whitelist, (
                f"tree referenced field outside whitelist: {s}"
            )
