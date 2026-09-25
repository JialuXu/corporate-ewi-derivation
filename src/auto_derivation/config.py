"""Project configuration.

Sources are layered (later wins):
1. defaults below
2. `.env` (gitignored — never commit it; see `.env.example` for the template)
3. `.env.local` (gitignored; put your real `AD_LLM_API_KEY` here)
4. process env vars prefixed `AD_`

Both `.env` and `.env.local` are gitignored. The only committed file is
`.env.example`, which carries keys with empty values. Do not `git add -f`
either dotenv file — that is how credentials reach a public repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DuplicatePolicy = Literal["strict", "warn", "max"]

REGISTRY_DIR = PROJECT_ROOT / "data" / "registry"
# Publishable, institution-neutral inventory (76 rows) — committed, and the
# fallback that makes a fresh clone runnable.
EXAMPLE_REGISTRY_CSV = REGISTRY_DIR / "metrics.example.csv"
# The full atomic-metric inventory is institution-internal (production table
# schema) and is deliberately NOT in this repo. Put your private copy here
# (gitignored) or point `AD_REGISTRY_CSV` at it anywhere on disk.
PRIVATE_REGISTRY_CSV = REGISTRY_DIR / "metrics.csv"


def _default_registry_csv() -> Path:
    """Private inventory if present, else the committed example."""
    return PRIVATE_REGISTRY_CSV if PRIVATE_REGISTRY_CSV.exists() else EXAMPLE_REGISTRY_CSV


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AD_",
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    # Override with AD_REGISTRY_CSV to load the full private inventory.
    registry_csv: Path = Field(default_factory=_default_registry_csv)
    synthetic_dir: Path = PROJECT_ROOT / "data" / "synthetic"
    # P1 persistence: per-run JSON / content-addressed explanation cards
    runs_dir: Path = PROJECT_ROOT / "data" / "runs"
    cards_dir: Path = PROJECT_ROOT / "data" / "cards"
    # Cross-batch analysis ledger (parquet, Hive partition by industry) +
    # industry knowledge corpus (TF-IDF over markdown).
    ledger_dir: Path = PROJECT_ROOT / "data" / "analysis_store" / "ledger"
    corpus_dir: Path = Path(__file__).resolve().parent / "knowledge" / "corpus"

    random_seed: int = 20260429
    n_synthetic_customers: int = 500
    n_synthetic_months: int = 24

    # GP / typecheck defaults
    max_tree_depth: int = 5
    max_tree_leaves: int = 8

    # Panel duplicate handling. `(entity_id, observation_date, metric_id)` is
    # supposed to be unique; the legacy PIVOT used max(value), which silently
    # collapsed real ETL bugs (重报 / 口径切换 / 手工补录).
    #   strict — raise on any duplicate (default; production-safe)
    #   warn   — log + write the dirty rows, then dedupe via max(value)
    #   max    — legacy behaviour (silent max), kept for migration only
    panel_duplicate_policy: DuplicatePolicy = "strict"

    # LLM provider — any OpenAI-compatible endpoint (OpenAI, DeepSeek,
    # Moonshot, vLLM, …). Defaults wired for DeepSeek.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-pro"

    @property
    def panel_path(self) -> Path:
        return self.synthetic_dir / "panel.parquet"

    @property
    def labels_path(self) -> Path:
        return self.synthetic_dir / "labels.parquet"

    @property
    def using_example_registry(self) -> bool:
        """True when running on the committed demo inventory rather than a
        private one — surfaced in logs so demo runs aren't mistaken for real."""
        return self.registry_csv.resolve() == EXAMPLE_REGISTRY_CSV.resolve()

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Fresh instance re-reading env vars / .env files at call time.

        Prefer this in tests and library code that needs custom values —
        pass `overrides` instead of monkeypatching the module-level
        `settings` singleton (which is constructed once at import)."""
        return cls(**overrides)


settings = Settings()
