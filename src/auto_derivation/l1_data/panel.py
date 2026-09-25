"""Entity panel I/O.

Physical layout: a single Parquet file (or partitioned dataset) with the
**long-form** schema:

    entity_id: string        — customer / 法人主体编号
    observation_date: date   — month-end snapshot date
    industry: string         — industry tag (used by peer operators)
    metric_id: string        — references MetricRegistry
    value: float64           — numeric coercion of the metric value (categorical
                               metrics are excluded from Phase 0; see
                               `write_long_panel` for the contract check)

For evaluation we **pivot to a wide panel** (one column per metric_id) using
DuckDB, so that L2 operators can compose `pl.Expr` over named columns. The
pivot happens inside `wide_panel()`; downstream code stays in Polars.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import duckdb
import polars as pl

from auto_derivation.config import DuplicatePolicy, settings

PANEL_COLUMNS = ["entity_id", "observation_date", "industry", "metric_id", "value"]

logger = logging.getLogger(__name__)


class PanelDuplicateError(ValueError):
    """Raised when wide_panel finds duplicate (entity, date, metric_id) rows
    under the strict policy."""


def write_long_panel(df: pl.DataFrame, path: Path) -> None:
    """Write a long-form panel to Parquet, validating schema + L1 type contract.

    The contract check (L1 type system support) ensures we don't silently
    accept CATEGORY/TEXT/DATE metrics that the current single `value: float64`
    schema cannot carry — those metrics would otherwise sail through to
    `wide_panel` and disappear from GP search without notice.
    """
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Long panel missing columns: {missing}")

    # Lazy import keeps panel.py independent of the registry at module-load time.
    from auto_derivation.l1_data.registry import default_registry
    from auto_derivation.l1_data.types import LogicalType

    registry = default_registry()
    numeric_types = {
        LogicalType.NUMERIC, LogicalType.INT, LogicalType.RATIO, LogicalType.BOOL,
    }
    seen_ids = df["metric_id"].unique().to_list()
    unsupported: list[tuple[str, str]] = []
    for mid in seen_ids:
        try:
            meta = registry.get(mid)
        except KeyError:
            # Unknown metric_id is a separate concern — caller may be testing
            # with synthetic ids. Surface it but don't block writing.
            logger.warning("metric_id %r not in registry; written as-is", mid)
            continue
        if meta.logical_type not in numeric_types:
            unsupported.append((mid, meta.dtype.value))
    if unsupported:
        raise ValueError(
            f"Cannot write long panel: {len(unsupported)} metric(s) have "
            f"non-numeric dtypes which the current single-float64 physical "
            f"schema cannot carry (see panel.py P2b: union schema). "
            f"Examples (metric_id, dtype): {unsupported[:5]}"
        )

    df.select(PANEL_COLUMNS).write_parquet(path)


def read_long_panel(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).select(PANEL_COLUMNS)


def wide_panel(
    panel_path: Path,
    metric_ids: Iterable[str] | None = None,
    *,
    duplicate_policy: DuplicatePolicy | None = None,
    dirty_rows_path: Path | None = None,
) -> pl.DataFrame:
    """Pivot the long panel to wide form, one column per metric_id.

    DuckDB does the pivot (efficient even at millions of rows); we then hand
    the result back to Polars for downstream expression evaluation.

    Duplicate rows (same `(entity_id, observation_date, metric_id)`) are
    handled per `duplicate_policy` (defaults to `Settings.panel_duplicate_policy`):

    - `strict`: raise `PanelDuplicateError` with the offending key samples
    - `warn`:   log + write full dirty rows to `dirty_rows_path` (if given),
                then continue with `max(value)` aggregation
    - `max`:    legacy silent behaviour (kept for migration parity)

    If `metric_ids` is None, all metrics in the panel are pivoted.
    """
    policy: DuplicatePolicy = duplicate_policy or settings.panel_duplicate_policy

    # Embed the parquet path directly in SQL via `read_parquet(...)` instead
    # of relying on DuckDB's Python-local variable replacement scan — the scan
    # only sees locals of the frame that calls `con.execute`, which would
    # break when we delegate to `_check_duplicates`.
    parquet_literal = str(panel_path).replace("'", "''")
    if metric_ids is not None:
        ids = ", ".join(f"'{m}'" for m in metric_ids)
        where_clause = f"WHERE metric_id IN ({ids})"
    else:
        where_clause = ""
    sql = f"""
        SELECT entity_id, observation_date, industry, metric_id, value
        FROM read_parquet('{parquet_literal}')
        {where_clause}
    """

    con = duckdb.connect()
    try:
        if policy != "max":
            _check_duplicates(con, sql, policy=policy, dirty_rows_path=dirty_rows_path)

        # PIVOT_WIDER turns metric_id values into columns.
        pivot_sql = f"""
            PIVOT ({sql})
            ON metric_id
            USING max(value)
            GROUP BY entity_id, observation_date, industry
        """
        arrow_tbl = con.execute(pivot_sql).arrow()
    finally:
        con.close()
    return pl.from_arrow(arrow_tbl).sort(["entity_id", "observation_date"])  # type: ignore[return-value]


def _check_duplicates(
    con: duckdb.DuckDBPyConnection,
    base_sql: str,
    *,
    policy: DuplicatePolicy,
    dirty_rows_path: Path | None,
) -> None:
    """Find duplicate (entity_id, observation_date, metric_id) rows.

    Under `strict`, raise with the first 5 duplicate keys. Under `warn`,
    log the count and (if `dirty_rows_path` is set) dump the full duplicate
    rows so operations can investigate.
    """
    dup_sql = f"""
        SELECT entity_id, observation_date, metric_id, COUNT(*) AS n
        FROM ({base_sql})
        GROUP BY entity_id, observation_date, metric_id
        HAVING n > 1
    """
    dup_rows = con.execute(dup_sql).fetchall()
    n_dup_keys = len(dup_rows)
    if n_dup_keys == 0:
        return

    samples = [
        {"entity_id": r[0], "observation_date": r[1], "metric_id": r[2], "n": r[3]}
        for r in dup_rows[:5]
    ]

    if policy == "strict":
        raise PanelDuplicateError(
            f"Found {n_dup_keys} duplicate (entity_id, observation_date, "
            f"metric_id) key(s) in panel; first 5: {samples}. "
            f"Set Settings.panel_duplicate_policy='warn' to log and continue, "
            f"or 'max' to keep the legacy silent-max behaviour."
        )

    # policy == "warn"
    logger.warning(
        "Panel has %d duplicate (entity_id, observation_date, metric_id) keys; "
        "proceeding with max(value) aggregation. Sample keys: %s",
        n_dup_keys, samples,
    )
    if dirty_rows_path is not None:
        dirty_full_sql = f"""
            SELECT t.*
            FROM ({base_sql}) t
            JOIN ({dup_sql}) d
              USING (entity_id, observation_date, metric_id)
        """
        dirty_tbl = con.execute(dirty_full_sql).to_arrow_table()
        dirty_rows_path.parent.mkdir(parents=True, exist_ok=True)
        pl.from_arrow(dirty_tbl).write_parquet(dirty_rows_path)  # type: ignore[union-attr]
        logger.warning("Wrote %d duplicate rows to %s", dirty_tbl.num_rows, dirty_rows_path)


def join_labels(panel: pl.DataFrame, labels_path: Path, label_name: str) -> pl.DataFrame:
    """Inner-join a label column onto a wide panel."""
    labels = pl.read_parquet(labels_path).filter(pl.col("label_name") == label_name)
    return panel.join(
        labels.select(["entity_id", "observation_date", "label_value"]),
        on=["entity_id", "observation_date"],
        how="inner",
    ).rename({"label_value": label_name})
