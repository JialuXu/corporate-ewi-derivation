"""Tests for `evaluation.splits.time_split` — P0.2 minimum-sample assertion.

Pre-fix, `time_split` would silently produce empty / overlapping slices when
the panel had too few distinct dates. Now it raises with a precise message.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from auto_derivation.evaluation.splits import time_split


def _panel(n_dates: int) -> pl.DataFrame:
    """Build a tiny panel with `n_dates` distinct observation_dates."""
    base = date(2025, 1, 31)
    schema = {"entity_id": pl.Utf8, "observation_date": pl.Date, "v": pl.Float64}
    if n_dates == 0:
        return pl.DataFrame(schema=schema)
    rows = [
        {"entity_id": "E1", "observation_date": base + timedelta(days=30 * i), "v": float(i)}
        for i in range(n_dates)
    ]
    return pl.DataFrame(rows, schema=schema)


def test_time_split_normal_case_ok():
    df = _panel(10)
    train, valid, oot, _dates = time_split(df)
    assert train.height >= 1 and valid.height >= 1 and oot.height >= 1


def test_time_split_empty_panel_raises():
    df = _panel(0)
    with pytest.raises(ValueError, match="no values"):
        time_split(df)


def test_time_split_too_few_dates_raises():
    """With default 0.6/0.2 fractions, n=2 cannot yield three non-empty slices.
    Pre-fix this returned valid/oot empty without complaint."""
    df = _panel(2)
    with pytest.raises(ValueError, match="cannot produce three non-empty slices"):
        time_split(df)


def test_time_split_minimum_n_for_default_fractions():
    """n=3 is the smallest size that works with default fractions."""
    df = _panel(3)
    train, valid, oot, _ = time_split(df)
    assert train.height == 1
    assert valid.height == 1
    assert oot.height == 1


def test_time_split_one_date_raises():
    df = _panel(1)
    with pytest.raises(ValueError, match="cannot produce three non-empty slices"):
        time_split(df)


def test_time_split_invalid_fractions():
    df = _panel(10)
    with pytest.raises(ValueError, match="must sum to"):
        time_split(df, train_frac=0.7, valid_frac=0.5)
    with pytest.raises(ValueError, match="must sum to"):
        time_split(df, train_frac=0.0, valid_frac=0.5)
