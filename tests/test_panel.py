"""Tests for `l1_data.panel` — duplicate handling + L1 type contract.

Plan refs:
- P0.1: `wide_panel` must surface duplicate (entity, date, metric) keys.
- P0.3: `write_long_panel` must reject metrics whose dtype the current
        single-float64 schema cannot carry.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from auto_derivation.l1_data.panel import (
    PanelDuplicateError,
    wide_panel,
    write_long_panel,
)


def _row(entity: str, d: date, metric_id: str, value: float, industry: str = "制造") -> dict:
    return {
        "entity_id": entity,
        "observation_date": d,
        "industry": industry,
        "metric_id": metric_id,
        "value": value,
    }


def _write_long(tmp_path, rows: list[dict]):
    df = pl.DataFrame(rows).with_columns(pl.col("observation_date").cast(pl.Date))
    path = tmp_path / "panel.parquet"
    write_long_panel(df, path)
    return path


# ---------- P0.3: write_long_panel L1 type contract ----------


def test_write_long_panel_accepts_numeric_metrics(tmp_path):
    """LOAN_0003 合同总金额 (AMOUNT) is numeric → should write fine."""
    path = _write_long(tmp_path, [_row("E1", date(2025, 1, 31), "LOAN_0003", 1.0)])
    assert path.exists()


def test_write_long_panel_rejects_category_metric(tmp_path):
    """LOAN_0002 客户行业大类 has dtype=分类 (CATEGORY) — single-float64
    schema can't carry it, so writing must fail loudly (P0.3)."""
    with pytest.raises(ValueError, match="non-numeric dtypes"):
        _write_long(tmp_path, [_row("E1", date(2025, 1, 31), "LOAN_0002", 0.0)])


def test_write_long_panel_rejects_date_metric(tmp_path):
    """LOAN_0004 合同起始日 has dtype=日期 (DATE)."""
    with pytest.raises(ValueError, match="non-numeric dtypes"):
        _write_long(tmp_path, [_row("E1", date(2025, 1, 31), "LOAN_0004", 0.0)])


def test_write_long_panel_rejects_text_metric(tmp_path):
    """LOAN_0001 客户所处地区 has dtype=文本 (TEXT)."""
    with pytest.raises(ValueError, match="non-numeric dtypes"):
        _write_long(tmp_path, [_row("E1", date(2025, 1, 31), "LOAN_0001", 0.0)])


# ---------- P0.1: wide_panel duplicate handling ----------


def _dup_panel(tmp_path):
    """Build a long panel that has two values for (E1, 2025-01-31, LOAN_0003)."""
    rows = [
        _row("E1", date(2025, 1, 31), "LOAN_0003", 100.0),
        _row("E1", date(2025, 1, 31), "LOAN_0003", 200.0),  # duplicate
        _row("E2", date(2025, 1, 31), "LOAN_0003", 300.0),
    ]
    return _write_long(tmp_path, rows)


def test_wide_panel_strict_raises_on_duplicate(tmp_path):
    path = _dup_panel(tmp_path)
    with pytest.raises(PanelDuplicateError, match="duplicate"):
        wide_panel(path, duplicate_policy="strict")


def test_wide_panel_warn_continues_and_writes_dirty_rows(tmp_path, caplog):
    path = _dup_panel(tmp_path)
    dirty_path = tmp_path / "dirty.parquet"
    with caplog.at_level("WARNING"):
        df = wide_panel(path, duplicate_policy="warn", dirty_rows_path=dirty_path)
    # Pivot still produces a wide row; max(value) wins for the dup.
    assert df.height == 2  # E1 + E2
    e1 = df.filter(pl.col("entity_id") == "E1")
    assert e1["LOAN_0003"].item() == 200.0  # max of 100 / 200
    # dirty rows file written, contains both duplicate originals
    assert dirty_path.exists()
    dirty_df = pl.read_parquet(dirty_path)
    assert dirty_df.filter(pl.col("entity_id") == "E1").height == 2
    assert any("duplicate" in r.message.lower() for r in caplog.records)


def test_wide_panel_max_silently_dedups(tmp_path):
    """Legacy parity: 'max' policy keeps the original silent behaviour."""
    path = _dup_panel(tmp_path)
    df = wide_panel(path, duplicate_policy="max")
    assert df.height == 2
    e1 = df.filter(pl.col("entity_id") == "E1")
    assert e1["LOAN_0003"].item() == 200.0


def test_wide_panel_no_duplicates_works(tmp_path):
    """Regression: clean panel goes through strict mode untouched."""
    rows = [
        _row("E1", date(2025, 1, 31), "LOAN_0003", 100.0),
        _row("E2", date(2025, 1, 31), "LOAN_0003", 300.0),
    ]
    path = _write_long(tmp_path, rows)
    df = wide_panel(path, duplicate_policy="strict")
    assert df.height == 2
    assert set(df["entity_id"].to_list()) == {"E1", "E2"}


def test_generate_rejects_short_horizon(tmp_path):
    """The injected bad-customer cascade spans months 14-22, so shorter
    horizons are rejected up front."""
    from auto_derivation.synthetic import generate

    with pytest.raises(ValueError, match="n_months"):
        generate(tmp_path, n_customers=10, n_months=18, seed=0)
