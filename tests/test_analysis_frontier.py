"""Tests for the global cross-batch Pareto frontier."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from auto_derivation.analysis.frontier import (
    GlobalFrontier,
    _non_dominated,
    render_frontier,
)
from auto_derivation.analysis.ledger import ExperimentRun, ParetoEntry, RunLedger


def _entry(expr: str, fitness: list[float], rank: int = 0) -> ParetoEntry:
    return ParetoEntry(
        expr_json=f'{{"kind":"field","name":"{expr}"}}',
        expr_sexpr=expr,
        fitness=fitness,
        rank=rank,
        crowding=0.0,
    )


def _run(industry: str | None, run_id: str, entries: list[ParetoEntry],
         panel_signature: str = "") -> ExperimentRun:
    return ExperimentRun(
        run_id=run_id,
        timestamp=datetime(2026, 5, 18, 10, 30, int(run_id[-2:], 16) % 60),
        label="Y_npl_12m",
        industry=industry,
        seed=0,
        panel_signature=panel_signature,
        pareto_entries=entries,
        history=[],
    )


def test_non_dominated_filters_strictly_dominated():
    a = _entry("a", [0.5, 0.5, 0.5, 0.5, -3.0, 0.0])
    b = _entry("b", [0.6, 0.6, 0.5, 0.5, -3.0, 0.0])   # dominates a in 2 dims
    c = _entry("c", [0.3, 0.9, 0.5, 0.5, -3.0, 0.0])   # incomparable to b
    result = _non_dominated([a, b, c])
    sexprs = {e.expr_sexpr for e in result}
    assert "a" not in sexprs
    assert "b" in sexprs
    assert "c" in sexprs


def test_non_dominated_keeps_all_when_all_incomparable():
    a = _entry("a", [0.5, 0.9, 0.3, 0.5, -3.0, 0.0])
    b = _entry("b", [0.9, 0.5, 0.5, 0.3, -3.0, 0.0])
    c = _entry("c", [0.3, 0.3, 0.9, 0.9, -3.0, 0.0])
    result = _non_dominated([a, b, c])
    assert len(result) == 3


def test_front_for_returns_only_industry_slice(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    ledger.append(_run("制造", "manu001a",
                       [_entry("a", [0.5, 0.5, 0.5, 0.5, -3.0, 0.0])]))
    ledger.append(_run("批零", "rtai001b",
                       [_entry("b", [0.7, 0.7, 0.7, 0.7, -3.0, 0.0])]))
    f = GlobalFrontier(ledger)
    manu = f.front_for("制造")
    assert len(manu) == 1
    assert manu[0].expr_sexpr == "a"


def test_is_dominated_uses_global_frontier(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    ledger.append(_run("制造", "manu001c",
                       [_entry("strong", [0.8, 0.8, 0.8, 0.8, -3.0, 0.0])]))
    f = GlobalFrontier(ledger)
    # Weaker candidate → dominated.
    assert f.is_dominated([0.5, 0.5, 0.5, 0.5, -3.0, 0.0], "制造") is True
    # Incomparable candidate (better in one dim, worse in another).
    assert f.is_dominated([0.9, 0.1, 0.5, 0.5, -3.0, 0.0], "制造") is False


def test_expansion_delta_counts_new_vs_displaced(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    old_run = _run("制造", "old00001",
                   [_entry("weak", [0.3, 0.3, 0.3, 0.3, -3.0, 0.0])],
                   panel_signature="sigA")
    ledger.append(old_run)
    new_run = _run("制造", "new00001",
                   [_entry("strong", [0.8, 0.8, 0.8, 0.8, -3.0, 0.0])],
                   panel_signature="sigA")
    ledger.append(new_run)

    f = GlobalFrontier(ledger)
    delta = f.expansion_delta(new_run)
    assert delta.new_on_front == 1
    assert delta.existing_dominated == 1
    assert delta.net_added == 0   # 1 added, 1 displaced
    assert delta.panel_signature_consistent is True


def test_expansion_delta_signature_inconsistency(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    ledger.append(_run("制造", "old00002",
                       [_entry("x", [0.5, 0.5, 0.5, 0.5, -3.0, 0.0])],
                       panel_signature="sigA"))
    ledger.append(_run("制造", "old00003",
                       [_entry("y", [0.6, 0.4, 0.5, 0.5, -3.0, 0.0])],
                       panel_signature="sigA"))
    new_run = _run("制造", "new00002",
                   [_entry("z", [0.9, 0.9, 0.9, 0.9, -3.0, 0.0])],
                   panel_signature="sigB")  # different signature!
    ledger.append(new_run)

    f = GlobalFrontier(ledger)
    delta = f.expansion_delta(new_run)
    assert delta.panel_signature_consistent is False


def test_render_frontier_empty_ledger(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    f = GlobalFrontier(ledger)
    md = render_frontier(f, "制造")
    assert "全局 Pareto 前沿" in md
    assert "制造" in md
    assert "当前没有" in md


def test_render_frontier_markdown_table(tmp_path: Path):
    ledger = RunLedger(tmp_path)
    ledger.append(_run("制造", "render01",
                       [_entry("(PctChange 营业收入 1)",
                               [0.45, 0.55, 0.8, 0.7, -3.0, -0.1])]))
    f = GlobalFrontier(ledger)
    md = render_frontier(f, "制造")
    assert "| 1 | 0.450 | 0.550 | 0.800 | 0.700 | 3 | 0.100 |" in md
    assert "PctChange" in md
