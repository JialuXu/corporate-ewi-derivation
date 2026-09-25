"""Time-based train/valid/oot splits.

DESIGN.md §5.4 + §8 pitfall 3:
- splits must be strictly time-ordered (no random shuffling)
- the OOT slice must postdate everything used to train
- features may need a T-3M lag if the warning target is "提前 3 个月发现"
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class SplitDates:
    train_end: date
    valid_end: date
    oot_end: date


def time_split(
    df: pl.DataFrame,
    *,
    date_col: str = "observation_date",
    train_frac: float = 0.6,
    valid_frac: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, SplitDates]:
    """Split by observation_date quantiles into train/valid/oot.

    Returns (train_df, valid_df, oot_df, dates). Raises ValueError if the
    panel does not have enough distinct dates to populate all three slices —
    the legacy code silently produced empty/overlapping slices in that case.
    """
    if not 0 < train_frac < 1 or not 0 < valid_frac < 1 or train_frac + valid_frac >= 1:
        raise ValueError("train_frac and valid_frac must sum to < 1 and both be in (0, 1)")

    sorted_dates = df.select(pl.col(date_col).unique().sort()).to_series()
    n = sorted_dates.len()
    if n == 0:
        raise ValueError(f"time_split: {date_col!r} has no values")

    train_idx = int(n * train_frac)
    valid_idx = int(n * (train_frac + valid_frac))

    # We need:
    #   train_idx >= 1     so train is non-empty
    #   valid_idx > train_idx     so valid is non-empty
    #   valid_idx < n             so oot is non-empty
    if not (train_idx >= 1 and train_idx < valid_idx < n):
        smallest_frac = min(train_frac, valid_frac, 1 - train_frac - valid_frac)
        suggested_n = int(1 / smallest_frac) + 1
        raise ValueError(
            f"time_split: cannot produce three non-empty slices from "
            f"n_dates={n} with train_frac={train_frac}, valid_frac={valid_frac} "
            f"(computed train_idx={train_idx}, valid_idx={valid_idx}). "
            f"Need at least ~{suggested_n} distinct dates, or adjust the fractions."
        )

    train_end = sorted_dates[train_idx - 1]
    valid_end = sorted_dates[valid_idx - 1]
    oot_end = sorted_dates[-1]

    train = df.filter(pl.col(date_col) <= train_end)
    valid = df.filter((pl.col(date_col) > train_end) & (pl.col(date_col) <= valid_end))
    oot = df.filter(pl.col(date_col) > valid_end)

    return train, valid, oot, SplitDates(train_end=train_end, valid_end=valid_end, oot_end=oot_end)


def lag_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    months: int,
    *,
    entity_col: str = "entity_id",
    date_col: str = "observation_date",
) -> pl.DataFrame:
    """Lag feature columns by `months` rows within each entity.

    Assumes the panel is monthly and sorted by (entity_id, observation_date).
    """
    return df.sort([entity_col, date_col]).with_columns(
        [pl.col(c).shift(months).over(entity_col).alias(c) for c in feature_cols]
    )
