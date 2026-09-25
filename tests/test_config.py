"""Settings tests — `from_env()` factory vs the module-level singleton."""
from __future__ import annotations

from auto_derivation.config import Settings, settings


def test_from_env_overrides_without_touching_singleton():
    fresh = Settings.from_env(llm_model="test-model", random_seed=7)
    assert fresh.llm_model == "test-model"
    assert fresh.random_seed == 7
    assert fresh is not settings
    assert settings.llm_model != "test-model"


def test_from_env_rereads_process_env(monkeypatch):
    monkeypatch.setenv("AD_LLM_MODEL", "env-model")
    assert Settings.from_env().llm_model == "env-model"
    # The import-time singleton keeps its original value.
    assert settings.llm_model != "env-model"
