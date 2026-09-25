"""Label builders.

Each builder returns a long-form DataFrame:
    entity_id, observation_date, label_name, label_value (Int8: 0/1)

Labels are forward-looking: at observation_date `t`, label_value = 1 iff a
qualifying event occurs in the window (t, t + horizon].
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class LabelSpec:
    name: str
    horizon_months: int
    description: str


LABEL_SPECS: dict[str, LabelSpec] = {
    "Y_overdue_3m": LabelSpec(
        name="Y_overdue_3m",
        horizon_months=3,
        description="未来 3 个月内逾期 ≥ 30 天",
    ),
    "Y_downgrade_6m": LabelSpec(
        name="Y_downgrade_6m",
        horizon_months=6,
        description="未来 6 个月内五级分类下迁",
    ),
    "Y_npl_12m": LabelSpec(
        name="Y_npl_12m",
        horizon_months=12,
        description="未来 12 个月内进入不良",
    ),
    "Y_early_signal": LabelSpec(
        name="Y_early_signal",
        horizon_months=6,
        description="未来 6 个月内出现司法/征信不利变化",
    ),
}


def build_labels(events: pl.DataFrame, observation_dates: pl.Series) -> pl.DataFrame:
    """Build all four labels from a per-customer event log.

    Parameters
    ----------
    events : DataFrame with columns
        entity_id, event_date (Date), event_type (str ∈ {overdue30, downgrade,
        npl, judicial_adverse, credit_adverse})
    observation_dates : Series of month-end snapshot dates

    Returns
    -------
    DataFrame[entity_id, observation_date, label_name, label_value]
    """
    out: list[pl.DataFrame] = []
    obs_df = pl.DataFrame({"observation_date": observation_dates}).with_columns(
        pl.col("observation_date").cast(pl.Date)
    )
    entities = events.select("entity_id").unique()
    grid = entities.join(obs_df, how="cross")

    event_to_labels: dict[str, list[str]] = {
        "overdue30": ["Y_overdue_3m"],
        "downgrade": ["Y_downgrade_6m"],
        "npl": ["Y_npl_12m"],
        "judicial_adverse": ["Y_early_signal"],
        "credit_adverse": ["Y_early_signal"],
    }

    for label_name, spec in LABEL_SPECS.items():
        relevant_events = [k for k, v in event_to_labels.items() if label_name in v]
        ev = events.filter(pl.col("event_type").is_in(relevant_events))

        joined = grid.join(ev, on="entity_id", how="left")
        horizon_days = spec.horizon_months * 31  # generous month-length

        flagged = (
            joined.with_columns(
                in_window=(
                    pl.col("event_date").is_not_null()
                    & (pl.col("event_date") > pl.col("observation_date"))
                    & (
                        pl.col("event_date")
                        <= pl.col("observation_date").dt.offset_by(f"{horizon_days}d")
                    )
                )
            )
            .group_by(["entity_id", "observation_date"])
            .agg(label_value=pl.col("in_window").any().cast(pl.Int8))
            .with_columns(label_name=pl.lit(label_name))
        )
        out.append(flagged)

    return pl.concat(out).select(
        ["entity_id", "observation_date", "label_name", "label_value"]
    )
