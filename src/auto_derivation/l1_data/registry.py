"""MetricRegistry — load the atomic-metric inventory CSV into a typed,
queryable structure.

Two inventories exist:
- `data/registry/metrics.example.csv` — committed, institution-neutral, 76 rows.
- the full private inventory — production table schema, never committed.
  Supply it via `AD_REGISTRY_CSV` or drop it at `data/registry/metrics.csv`.

Used by:
- L2 operators (resolve a metric_id → dtype, time_grain when type-checking)
- L4 LLM explanation (field_metadata for prompt construction)
- CLI (`derive registry list`)
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .types import (
    DTYPE_TO_LOGICAL,
    Dtype,
    LogicalType,
    NullSemantics,
    TimeGrain,
)


@dataclass(frozen=True, slots=True)
class MetricMeta:
    metric_id: str
    source: str
    name_cn: str
    dtype: Dtype
    time_grain: TimeGrain
    update_freq: str
    value_range: str
    business_tag: str
    null_semantics: NullSemantics

    @property
    def logical_type(self) -> LogicalType:
        return DTYPE_TO_LOGICAL[self.dtype]


class MetricRegistry:
    """In-memory registry built from 原子指标清单.csv.

    Lookups are by `metric_id` (primary), `name_cn`, or filtered by source/dtype.
    """

    def __init__(self, metrics: Iterable[MetricMeta]):
        self._by_id: dict[str, MetricMeta] = {}
        self._by_name: dict[str, MetricMeta] = {}
        self._names_desc: list[str] | None = None
        for m in metrics:
            if m.metric_id in self._by_id:
                raise ValueError(f"Duplicate metric_id: {m.metric_id}")
            self._by_id[m.metric_id] = m
            # name collisions are allowed (different sources may share a name);
            # the *first* occurrence wins for name lookup
            self._by_name.setdefault(m.name_cn, m)

    # --- construction ---

    @classmethod
    def from_csv(cls, path: Path) -> MetricRegistry:
        if not path.exists():
            raise FileNotFoundError(
                f"Metric inventory not found: {path}\n"
                "The full inventory is institution-internal and is not shipped "
                "with this repo. Either set AD_REGISTRY_CSV to your private "
                "copy, place it at data/registry/metrics.csv, or fall back to "
                "the committed data/registry/metrics.example.csv."
            )
        rows: list[MetricMeta] = []
        # CSV is UTF-8 with BOM and contains Chinese full-width commas inside
        # value_range / null_semantics fields; csv.DictReader handles quoting,
        # but we must use utf-8-sig to strip the BOM.
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                rows.append(_parse_row(raw))
        return cls(rows)

    # --- queries ---

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def __contains__(self, metric_id: str) -> bool:
        return metric_id in self._by_id

    def get(self, metric_id: str) -> MetricMeta:
        try:
            return self._by_id[metric_id]
        except KeyError as e:
            raise KeyError(f"Unknown metric_id: {metric_id}") from e

    def get_by_name(self, name_cn: str) -> MetricMeta:
        try:
            return self._by_name[name_cn]
        except KeyError as e:
            raise KeyError(f"Unknown metric name: {name_cn}") from e

    def resolve(self, key: str) -> MetricMeta:
        """Resolve either a metric_id or a Chinese name."""
        if key in self._by_id:
            return self._by_id[key]
        if key in self._by_name:
            return self._by_name[key]
        raise KeyError(f"Unknown metric: {key}")

    def by_source(self, source: str) -> list[MetricMeta]:
        return [m for m in self._by_id.values() if m.source == source]

    def by_dtype(self, dtype: Dtype) -> list[MetricMeta]:
        return [m for m in self._by_id.values() if m.dtype == dtype]

    def by_business_tag(self, tag: str) -> list[MetricMeta]:
        return [m for m in self._by_id.values() if m.business_tag == tag]

    def names_cn_by_length_desc(self) -> list[str]:
        """All `name_cn`, longest first — for substring scans that must match
        "营业收入同比增长率" before its substring "营业收入". Cached."""
        if self._names_desc is None:
            self._names_desc = sorted(
                (n for n in self._by_name if n), key=len, reverse=True
            )
        return self._names_desc

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self._by_id.values():
            counts[m.source] = counts.get(m.source, 0) + 1
        return counts


def _parse_row(raw: dict[str, str]) -> MetricMeta:
    def _get(key: str) -> str:
        return (raw.get(key) or "").strip()

    dtype_raw = _get("dtype")
    try:
        dtype = Dtype(dtype_raw)
    except ValueError as e:
        raise ValueError(
            f"Unknown dtype {dtype_raw!r} for metric {_get('metric_id')!r}"
        ) from e

    grain_raw = _get("time_grain")
    try:
        grain = TimeGrain(grain_raw)
    except ValueError as e:
        raise ValueError(
            f"Unknown time_grain {grain_raw!r} for metric {_get('metric_id')!r}"
        ) from e

    return MetricMeta(
        metric_id=_get("metric_id"),
        source=_get("source"),
        name_cn=_get("name_cn"),
        dtype=dtype,
        time_grain=grain,
        update_freq=_get("update_freq"),
        value_range=_get("value_range"),
        business_tag=_get("business_tag"),
        null_semantics=NullSemantics.parse(_get("null_semantics")),
    )


logger = logging.getLogger(__name__)

_default_registry: MetricRegistry | None = None


def default_registry() -> MetricRegistry:
    """Lazy-loaded singleton built from the configured registry CSV."""
    global _default_registry
    if _default_registry is None:
        from auto_derivation.config import settings

        _default_registry = MetricRegistry.from_csv(settings.registry_csv)
        if settings.using_example_registry:
            logger.info(
                "Loaded the example inventory (%d metrics) — demo data, not a "
                "production inventory. Set AD_REGISTRY_CSV to use your own.",
                len(_default_registry),
            )
        else:
            logger.info(
                "Loaded metric inventory from %s (%d metrics).",
                settings.registry_csv, len(_default_registry),
            )
    return _default_registry


def reset_default_registry() -> None:
    """Drop the cached singleton — for tests that swap `settings.registry_csv`."""
    global _default_registry
    _default_registry = None
