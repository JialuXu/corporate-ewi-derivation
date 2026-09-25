"""Experiment ledger — append-only parquet.

Each GP run (one `(label, industry, seed)` triple) is written to its own
parquet file under `data/analysis_store/ledger/industry={industry}/`. One
file per run means no concurrent-write contention and no rewrite-on-append.

Layout::

    ledger/
    ├── industry=制造/
    │   ├── run_20260518T103045_a1b2c3d4.parquet
    │   └── run_20260518T104112_e5f6g7h8.parquet
    └── industry=__none__/
        └── run_20260518T110055_xxxxxxxx.parquet   # industry=None case

`__none__` is the Hive-partition sentinel for the "no industry filter" runs;
parquet partition values cannot be null.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from auto_derivation.expression.tree import to_json as expr_to_json
from auto_derivation.l3_search.fitness import INVALID_FITNESS
from auto_derivation.l3_search.individual import Individual
from auto_derivation.l3_search.search import SearchResult

from .schema import LEDGER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

INDUSTRY_NONE_SENTINEL = "__none__"


class ParetoEntry(BaseModel):
    """One Individual on the final Pareto front, serialised for the ledger."""

    expr_json: str
    expr_sexpr: str
    fitness: list[float]
    rank: int
    crowding: float


class ExperimentRun(BaseModel):
    """One GP search run.

    The `fitness` field on each ParetoEntry is `list[float]` (not a fixed
    tuple) so we can grow the dimension without breaking the parquet schema.
    """

    run_id: str
    timestamp: datetime
    label: str
    industry: str | None
    seed: int
    schema_version: int = Field(default=LEDGER_SCHEMA_VERSION)
    gp_config: dict[str, Any] = Field(default_factory=dict)
    fitness_config: dict[str, Any] = Field(default_factory=dict)
    panel_signature: str = ""
    registry_sha: str = ""
    git_sha: str | None = None
    pareto_entries: list[ParetoEntry] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)


def _industry_partition(industry: str | None) -> str:
    return industry if industry else INDUSTRY_NONE_SENTINEL


def _make_run_id() -> str:
    return uuid.uuid4().hex[:8]


def _panel_signature(panel_path: Path) -> str:
    """Fast signature: head 1000 rows + column names, sha256, first 16 bytes.

    NOT a content-hash of the whole file — pyarrow rewrites metadata on
    read/write and would change the full-file hash. This gives a "shape +
    head sample" fingerprint that is stable across reads."""
    if not panel_path.exists():
        return ""
    try:
        head = pl.scan_parquet(panel_path).head(1000).collect()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("panel_signature failed: %s", e)
        return ""
    h = hashlib.sha256()
    h.update(",".join(head.columns).encode("utf-8"))
    h.update(head.write_csv().encode("utf-8"))
    return h.hexdigest()[:16]


def _registry_sha(registry_csv: Path) -> str:
    if not registry_csv.exists():
        return ""
    h = hashlib.sha256()
    with registry_csv.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _coerce_config_to_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, BaseModel):
        return cfg.model_dump()
    return {"_repr": repr(cfg)}


def _individuals_to_entries(individuals: list[Individual]) -> list[ParetoEntry]:
    out: list[ParetoEntry] = []
    for ind in individuals:
        fit = ind.fitness
        if fit is None or fit == INVALID_FITNESS:
            continue
        out.append(
            ParetoEntry(
                expr_json=expr_to_json(ind.expr),
                expr_sexpr=ind.expr.to_sexpr(),
                fitness=list(fit),
                rank=ind.rank,
                crowding=float(ind.crowding),
            )
        )
    return out


class RunLedger:
    """Append-only parquet store.

    Each `append()` writes one new parquet file. Reads scan all files under
    `root/industry=*/*.parquet` and rebuild `ExperimentRun` objects via the
    pydantic schema.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    # --- write ---

    def append(self, run: ExperimentRun) -> Path:
        partition_dir = self.root / f"industry={_industry_partition(run.industry)}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        ts_compact = run.timestamp.strftime("%Y%m%dT%H%M%S")
        filename = f"run_{ts_compact}_{run.run_id}.parquet"
        path = partition_dir / filename
        # Flatten to all-scalar columns: nested objects (lists / dicts) go via
        # JSON strings. parquet handles strings unambiguously even when empty,
        # and downstream consumers read the JSON cells directly without struct schema.
        record = _flatten_for_parquet(run)
        df = pl.DataFrame([record])
        df.write_parquet(path)
        return path

    # --- read ---

    def _all_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("industry=*/*.parquet"), key=lambda p: p.stat().st_mtime)

    def load_all(self) -> list[ExperimentRun]:
        runs: list[ExperimentRun] = []
        for f in self._all_files():
            runs.extend(self._read_file(f))
        return runs

    def query(
        self,
        *,
        label: str | None = None,
        industry: str | None = None,
        since: datetime | None = None,
    ) -> list[ExperimentRun]:
        files: list[Path]
        if industry is not None:
            partition = self.root / f"industry={_industry_partition(industry)}"
            files = sorted(partition.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
        else:
            files = self._all_files()
        out: list[ExperimentRun] = []
        for f in files:
            for run in self._read_file(f):
                if label is not None and run.label != label:
                    continue
                if since is not None and run.timestamp < since:
                    continue
                out.append(run)
        return out

    def latest(self, n: int) -> list[ExperimentRun]:
        files = self._all_files()
        if not files:
            return []
        runs: list[ExperimentRun] = []
        for f in reversed(files):
            runs.extend(self._read_file(f))
            if len(runs) >= n:
                break
        return runs[:n]

    def _read_file(self, path: Path) -> list[ExperimentRun]:
        try:
            df = pl.read_parquet(path)
        except Exception as e:  # pragma: no cover
            logger.warning("ledger: failed to read %s: %s", path, e)
            return []
        out: list[ExperimentRun] = []
        for row in df.to_dicts():
            sv = row.get("schema_version") or 1
            if sv > LEDGER_SCHEMA_VERSION:
                logger.warning(
                    "ledger file %s has schema_version=%s newer than reader (%s)",
                    path, sv, LEDGER_SCHEMA_VERSION,
                )
            try:
                # polars may stringify nested structs; re-coerce.
                run = ExperimentRun.model_validate(_normalise_row(row))
            except Exception as e:
                logger.warning("ledger: invalid row in %s: %s", path, e)
                continue
            out.append(run)
        return out


_JSON_FIELDS = ("gp_config", "fitness_config", "pareto_entries", "history")


def _flatten_for_parquet(run: ExperimentRun) -> dict[str, Any]:
    """Convert ExperimentRun → all-scalar dict for parquet writing.

    Nested fields (gp_config, fitness_config, pareto_entries, history) are
    serialised to JSON strings. This sidesteps polars' refusal to write
    empty list-of-struct types and gives downstream consumers a flat schema.
    """
    record = run.model_dump(mode="json")
    for k in _JSON_FIELDS:
        v = record.get(k)
        record[k] = json.dumps(v if v is not None else [], ensure_ascii=False)
    return record


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """Revive JSON-stringified sub-fields and tolerate datetime variants."""
    out = dict(row)
    for k in _JSON_FIELDS:
        v = out.get(k)
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = [] if k in ("pareto_entries", "history") else {}
    return out


# --- wrapper: SearchResult → ExperimentRun + write ---


def record_search_result(
    res: SearchResult,
    *,
    label: str,
    industry: str | None,
    seed: int,
    gp_config: Any = None,
    fitness_config: Any = None,
    panel_path: Path | None = None,
    registry_csv: Path | None = None,
    git_sha: str | None = None,
    ledger: RunLedger,
) -> ExperimentRun:
    """Build an ExperimentRun from a SearchResult and append it to the ledger.

    Caller is expected to supply the same `gp_config` / `fitness_config`
    instances that were passed to `search()` — we snapshot them so the run
    is reproducible from the ledger alone.
    """
    run = ExperimentRun(
        run_id=_make_run_id(),
        timestamp=datetime.now(),
        label=label,
        industry=industry,
        seed=seed,
        schema_version=LEDGER_SCHEMA_VERSION,
        gp_config=_coerce_config_to_dict(gp_config),
        fitness_config=_coerce_config_to_dict(fitness_config),
        panel_signature=_panel_signature(panel_path) if panel_path else "",
        registry_sha=_registry_sha(registry_csv) if registry_csv else "",
        git_sha=git_sha,
        pareto_entries=_individuals_to_entries(res.pareto_front),
        history=list(res.history),
    )
    ledger.append(run)
    return run
