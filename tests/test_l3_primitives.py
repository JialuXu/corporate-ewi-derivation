"""Tests for `l3_search.primitives.TypedPrimitiveSet` — P0.3 visibility of
metrics that the executor's single-float64 schema cannot carry.

Pre-fix, CATEGORY/TEXT/DATE metrics were silently filtered out — the L1 type
system advertised support that the executor didn't deliver. Now the dropped
metrics are tracked on `unsupported_fields` so callers / logs can surface
the gap.
"""
from __future__ import annotations

from auto_derivation.l1_data.registry import default_registry
from auto_derivation.l1_data.types import LogicalType
from auto_derivation.l2_operators.registry import default_operator_registry
from auto_derivation.l3_search.primitives import TypedPrimitiveSet


def test_unsupported_fields_populated():
    primset = TypedPrimitiveSet(
        op_registry=default_operator_registry(),
        metric_registry=default_registry(),
    )
    # The registry contains plenty of CATEGORY/TEXT/DATE metrics; we expect
    # at least one of each ended up on `unsupported_fields`.
    types = {m.logical_type for m in primset.unsupported_fields}
    assert LogicalType.CATEGORY in types
    assert LogicalType.DATE in types


def test_unsupported_fields_excluded_from_pickable_pool():
    """A metric appearing in `unsupported_fields` must NOT also be available
    via `fields_of_type(...)` for its logical type."""
    primset = TypedPrimitiveSet(
        op_registry=default_operator_registry(),
        metric_registry=default_registry(),
    )
    pickable_ids = {m.metric_id for ms in primset._fields_by_type.values() for m in ms}
    unsupported_ids = {m.metric_id for m in primset.unsupported_fields}
    assert pickable_ids.isdisjoint(unsupported_ids)


def test_field_whitelist_filters_before_unsupported_check():
    """Whitelist takes precedence: a CATEGORY metric not in the whitelist
    should not even reach the unsupported list."""
    primset = TypedPrimitiveSet(
        op_registry=default_operator_registry(),
        metric_registry=default_registry(),
        field_whitelist=frozenset({"FIN_0001"}),  # numeric
    )
    assert primset.unsupported_fields == []
    pickable = {m.metric_id for ms in primset._fields_by_type.values() for m in ms}
    assert pickable == {"FIN_0001"}
