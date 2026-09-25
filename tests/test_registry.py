from itertools import pairwise

import pytest

from auto_derivation.l1_data.registry import MetricRegistry, default_registry
from auto_derivation.l1_data.types import Dtype, NullSemantics, TimeGrain


def test_registry_loads_example_inventory():
    # The suite is pinned to data/registry/metrics.example.csv by conftest;
    # the full private inventory is not shipped with this repo.
    reg = default_registry()
    assert len(reg) == 76


def test_registry_source_distribution():
    reg = default_registry()
    counts = reg.sources()
    # Sanity-check the per-source counts of the example inventory.
    assert counts["财务"] == 17
    assert counts["行内信贷"] == 12
    assert counts["资金流"] == 9


def test_missing_inventory_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError, match="AD_REGISTRY_CSV"):
        MetricRegistry.from_csv(missing)


def test_example_inventory_covers_synthetic_generator():
    """`gen-synthetic` writes a fixed metric_id set — the example inventory
    must keep carrying all of them, or the demo path breaks on a fresh clone."""
    from auto_derivation.synthetic import METRICS

    reg = default_registry()
    missing = [mid for mid in METRICS if mid not in reg]
    assert not missing, f"example inventory is missing: {missing}"


def test_example_metric_ids_are_sequential_per_prefix():
    """Ids run PREFIX_0001, PREFIX_0002, … in file order — gaps would hint at
    rows copied from a larger private inventory."""
    reg = default_registry()
    seen: dict[str, int] = {}
    for meta in reg:
        prefix, num = meta.metric_id.rsplit("_", 1)
        expected = seen.get(prefix, 0) + 1
        assert num == f"{expected:04d}", f"{meta.metric_id}: expected {prefix}_{expected:04d}"
        seen[prefix] = expected


def test_registry_resolve_by_name_and_id():
    reg = default_registry()
    by_id = reg.get("FIN_0001")
    by_name = reg.get_by_name("营业收入")
    assert by_id.metric_id == by_name.metric_id == "FIN_0001"
    assert by_id.dtype == Dtype.AMOUNT
    assert by_id.time_grain == TimeGrain.QUARTER


def test_registry_resolves_either_form():
    reg = default_registry()
    assert reg.resolve("FIN_0001").metric_id == "FIN_0001"
    assert reg.resolve("营业收入").metric_id == "FIN_0001"


def test_names_by_length_desc_sorted_and_cached():
    reg = default_registry()
    names = reg.names_cn_by_length_desc()
    assert names, "registry has no names"
    assert all(len(a) >= len(b) for a, b in pairwise(names))
    # Same list object on second call — computed once.
    assert names is reg.names_cn_by_length_desc()


def test_null_semantics_preserves_distinguish_marker():
    reg = default_registry()
    # LOAN_0001 客户所处地区 has the "未维护、未授权查询和不适用需区分" marker.
    meta = reg.get("LOAN_0001")
    assert meta.null_semantics == NullSemantics.DISTINGUISH
