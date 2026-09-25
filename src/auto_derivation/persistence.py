"""Run / card persistence — turn ephemeral results into durable artifacts.

Two layouts:

    data/runs/<utc_ts>__<label>__<scope>/
        pareto.json    list[{expr_dict, fitness, rank, crowding}]
        history.json   list[{gen, valid, n_fronts, front0_size, best_iv, best_ks}]
        config.json    {gp_config, fitness_config, git_sha, panel_sha256, label, scope, ts}
    data/runs/index.jsonl
        one line per run: {run_id, ts, label, scope, n_pareto, best_iv, best_ks}

    data/cards/<sha12[:2]>/<sha12>.json
        {expr_dict, expr_sexpr, label, card, consistency_warnings, ts}

`<sha12>` is the first 12 hex chars of sha256(canonical(expr_dict)) — content-
addressed so re-explaining the same expression idempotently overwrites.

`index.jsonl` is append-only and grep-able; promote it to a queryable store
once real multi-dim queries are needed (Phase 1+).
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from auto_derivation.config import settings
from auto_derivation.expression.tree import ExprNode

logger = logging.getLogger(__name__)


# ---------- helpers ----------


def _now_utc_ts() -> str:
    """Compact UTC timestamp safe for filenames."""
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")


def _expr_sha12(expr: ExprNode) -> str:
    return sha256(_canonical_json_bytes(expr.to_dict())).hexdigest()[:12]


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    """Best-effort git HEAD SHA — None if the workspace isn't a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(settings.project_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses / tuples / Paths into JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    # pydantic v2 BaseModel: dispatch via getattr so the narrowing doesn't
    # bleed in from the dataclass branch above.
    mdump = getattr(obj, "model_dump", None)
    if callable(mdump):
        return _to_jsonable(mdump())
    return str(obj)


# ---------- search runs ----------


def save_search_run(
    *,
    pareto: list[Any],          # list[Individual]; not typed to avoid circular import
    history: list[dict],
    label: str,
    scope: str | None,
    gp_config: Any,
    fitness_config: Any,
    panel_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write pareto/history/config triple + append index.jsonl. Returns run dir.

    `pareto` items must expose `.expr: ExprNode`, `.fitness: tuple[float,...] | None`,
    `.rank: int`, `.crowding: float` (i.e. the GP `Individual` dataclass).
    """
    base = out_dir or settings.runs_dir
    safe_scope = (scope or "all").replace("/", "_").replace(" ", "_")
    safe_label = label.replace("/", "_").replace(" ", "_")
    ts = _now_utc_ts()
    run_id = f"{ts}__{safe_label}__{safe_scope}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pareto_payload = [
        {
            "expr_dict": ind.expr.to_dict(),
            "expr_sexpr": ind.expr.to_sexpr(),
            "fitness": list(ind.fitness) if ind.fitness is not None else None,
            "rank": int(ind.rank),
            "crowding": float(ind.crowding),
        }
        for ind in pareto
    ]
    (run_dir / "pareto.json").write_text(
        json.dumps(pareto_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(_to_jsonable(history), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    panel_sha = _file_sha256(panel_path) if panel_path and panel_path.exists() else None
    config_payload = {
        "ts": ts,
        "label": label,
        "scope": scope,
        "git_sha": _git_sha(),
        "panel_path": str(panel_path) if panel_path else None,
        "panel_sha256": panel_sha,
        "gp_config": _to_jsonable(gp_config),
        "fitness_config": _to_jsonable(fitness_config),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Index: one-line summary for grep / quick listing.
    best_iv = max((p["fitness"][0] for p in pareto_payload if p["fitness"]), default=None)
    best_ks = max((p["fitness"][1] for p in pareto_payload if p["fitness"]), default=None)
    index_row = {
        "run_id": run_id,
        "ts": ts,
        "label": label,
        "scope": scope,
        "n_pareto": len(pareto_payload),
        "best_iv": best_iv,
        "best_ks": best_ks,
    }
    base.mkdir(parents=True, exist_ok=True)
    with (base / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_row, ensure_ascii=False) + "\n")

    logger.info("saved search run → %s (n_pareto=%d)", run_dir, len(pareto_payload))
    return run_dir


# ---------- explanation cards ----------


def save_explanation_card(
    *,
    expr: ExprNode,
    label: str,
    result: Any,           # ExplanationResult; not typed to avoid circular import
    out_dir: Path | None = None,
) -> Path:
    """Write the card under `data/cards/<sha12[:2]>/<sha12>.json`.

    Content-addressed: re-explaining the same `(expr_dict)` overwrites in place.
    `label` is recorded inside the file but does not split the filename — two
    explanations of the same expression for different labels share the path,
    on the assumption that explanation is fundamentally about the expression
    and the latest one wins.
    """
    base = out_dir or settings.cards_dir
    sha12 = _expr_sha12(expr)
    sub = base / sha12[:2]
    sub.mkdir(parents=True, exist_ok=True)
    target = sub / f"{sha12}.json"

    payload = {
        "ts": _now_utc_ts(),
        "expr_sha12": sha12,
        "expr_dict": expr.to_dict(),
        "expr_sexpr": expr.to_sexpr(),
        "label": label,
        "card": result.card.model_dump(),
        "consistency_warnings": list(getattr(result, "consistency_warnings", []) or []),
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("saved explanation card → %s", target)
    return target
