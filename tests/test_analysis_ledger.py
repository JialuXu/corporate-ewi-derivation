"""Tests for the cross-batch experiment ledger.

We hit two flavours of assertion:
1. Round-trip: write an ExperimentRun, read it back, fields match.
2. End-to-end: run a tiny GP search, wrap with `record_search_result`, then
   read the ledger and verify the pareto front matches.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from auto_derivation.analysis.ledger import (
    INDUSTRY_NONE_SENTINEL,
    ExperimentRun,
    ParetoEntry,
    RunLedger,
    record_search_result,
)
from auto_derivation.analysis.schema import LEDGER_SCHEMA_VERSION


def _make_run(industry: str | None = "制造", *, run_id: str = "abc12345") -> ExperimentRun:
    return ExperimentRun(
        run_id=run_id,
        timestamp=datetime(2026, 5, 18, 10, 30, 45),
        label="Y_npl_12m",
        industry=industry,
        seed=42,
        gp_config={"pop_size": 10, "n_gens": 2},
        fitness_config={"label_col": "Y_npl_12m"},
        panel_signature="abcdef0123456789",
        registry_sha="fedcba9876543210",
        git_sha="deadbeef",
        pareto_entries=[
            ParetoEntry(
                expr_json='{"kind":"field","name":"营业收入"}',
                expr_sexpr="营业收入",
                fitness=[0.3, 0.4, 0.8, 0.5, -3.0, 0.0],
                rank=0,
                crowding=1.5,
            ),
        ],
        history=[{"gen": 0, "valid": 5, "best_iv": 0.3}],
    )


def test_append_creates_parquet_under_hive_partition(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    run = _make_run(industry="制造", run_id="run00001")
    written = ledger.append(run)
    assert written.exists()
    assert written.parent.name == "industry=制造"
    assert written.name.startswith("run_20260518T103045_run00001")


def test_append_none_industry_uses_sentinel(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    run = _make_run(industry=None, run_id="run00002")
    written = ledger.append(run)
    assert written.parent.name == f"industry={INDUSTRY_NONE_SENTINEL}"


def test_round_trip_preserves_fields(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    original = _make_run(industry="制造", run_id="run00003")
    ledger.append(original)

    runs = ledger.load_all()
    assert len(runs) == 1
    back = runs[0]
    assert back.run_id == original.run_id
    assert back.label == original.label
    assert back.industry == original.industry
    assert back.seed == original.seed
    assert back.schema_version == LEDGER_SCHEMA_VERSION
    assert back.gp_config["pop_size"] == 10
    assert back.fitness_config["label_col"] == "Y_npl_12m"
    assert back.panel_signature == "abcdef0123456789"
    assert back.registry_sha == "fedcba9876543210"
    assert back.git_sha == "deadbeef"
    assert len(back.pareto_entries) == 1
    e = back.pareto_entries[0]
    assert e.expr_sexpr == "营业收入"
    assert e.fitness == [0.3, 0.4, 0.8, 0.5, -3.0, 0.0]
    assert e.rank == 0


def test_query_by_label_industry_since(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    a = _make_run(industry="制造", run_id="aaaa1111")
    b = _make_run(industry="批零", run_id="bbbb2222")
    b.label = "Y_overdue_3m"
    c = _make_run(industry="制造", run_id="cccc3333")
    c.timestamp = datetime(2026, 1, 1)
    ledger.append(a)
    ledger.append(b)
    ledger.append(c)

    # by industry only
    runs = ledger.query(industry="制造")
    ids = sorted(r.run_id for r in runs)
    assert ids == ["aaaa1111", "cccc3333"]

    # by label only
    runs = ledger.query(label="Y_overdue_3m")
    assert [r.run_id for r in runs] == ["bbbb2222"]

    # by since
    runs = ledger.query(since=datetime(2026, 5, 1))
    ids = sorted(r.run_id for r in runs)
    assert "cccc3333" not in ids   # older than cutoff
    assert "aaaa1111" in ids and "bbbb2222" in ids


def test_query_industry_none_filter(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    ledger.append(_make_run(industry=None, run_id="none0001"))
    ledger.append(_make_run(industry="制造", run_id="manu0001"))

    runs = ledger.query(industry=None)  # None means "no filter", returns all
    assert len(runs) == 2


def test_load_all_empty_returns_empty_list(tmp_path: Path):
    ledger = RunLedger(tmp_path / "nonexistent")
    assert ledger.load_all() == []
    assert ledger.query() == []


def test_latest_n_returns_most_recent(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    # mtime ordering, not timestamp field — we just write 3 in order.
    for i in range(3):
        r = _make_run(industry="制造", run_id=f"r{i:07d}")
        r.timestamp = datetime(2026, 5, 18, 10, 0, i)
        ledger.append(r)
    latest = ledger.latest(2)
    assert len(latest) == 2


def test_record_search_result_end_to_end(synthetic_paths, tmp_path):
    """Run a tiny GP search and verify the ledger captures the pareto front."""
    panel_path, labels_path = synthetic_paths
    from auto_derivation.l3_search.fitness import FitnessConfig
    from auto_derivation.l3_search.search import GPConfig, search

    cfg = GPConfig(pop_size=10, n_gens=2, seed=7)
    fc = FitnessConfig(label_col="Y_npl_12m", industry="制造")
    res = search(panel_path, labels_path, label="Y_npl_12m", industry="制造",
                 config=cfg, fitness_cfg=fc)

    ledger = RunLedger(tmp_path / "ledger")
    run = record_search_result(
        res, label="Y_npl_12m", industry="制造", seed=7,
        gp_config=cfg, fitness_config=fc,
        panel_path=panel_path, ledger=ledger,
    )
    assert run.label == "Y_npl_12m"
    assert run.industry == "制造"
    assert run.seed == 7
    assert run.gp_config["pop_size"] == 10
    # pareto_entries may be empty on a 10×2 run, but the ledger file should exist.
    files = list((tmp_path / "ledger" / "industry=制造").glob("*.parquet"))
    assert len(files) == 1

    # Reload from disk and verify it matches.
    reloaded = ledger.load_all()
    assert len(reloaded) == 1
    assert reloaded[0].run_id == run.run_id
    assert len(reloaded[0].pareto_entries) == len(run.pareto_entries)
    assert reloaded[0].panel_signature != ""   # signature was computed
