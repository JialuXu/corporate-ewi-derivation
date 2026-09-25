"""Tests for `OpenAICompatClient` retry / backoff — P1.3.

We don't actually call any LLM API. Instead we monkeypatch:
  - `_retriable_exceptions` so the test owns the retriable type
  - the lazy `from openai import OpenAI` import so the client driver returns
    a fake whose `chat.completions.create` we control
  - `time.sleep` so the test runs instantly while still recording the delays

The contract:
  - max_retries=N  →  up to N+1 total attempts
  - successful call returns immediately, no sleeps
  - retriable error sleeps base_delay * 2**attempt then retries
  - non-retriable error raises on first encounter (no retry, no sleep)
  - exhausted retries raise RuntimeError with `from last_err`
"""
from __future__ import annotations

import sys
import types

import pytest

from auto_derivation.l4_explain import llm_client as llm_client_mod
from auto_derivation.l4_explain.llm_client import OpenAICompatClient


class _RetriableError(Exception):
    """Stand-in for openai.RateLimitError; we register it as retriable for
    the duration of one test."""


class _NonRetriableError(Exception):
    pass


def _install_fake_openai(monkeypatch, *, behavior):
    """Make `from openai import OpenAI` return a class whose
    `.chat.completions.create(...)` defers to `behavior(call_idx)`.

    `behavior(i)` either returns a faked response object or raises.
    """
    calls = {"n": 0}

    class _FakeChoice:
        def __init__(self, content):
            self.message = types.SimpleNamespace(content=content)

    class _FakeResp:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def create(self, **kwargs):
            i = calls["n"]
            calls["n"] += 1
            result = behavior(i)
            return _FakeResp(result)

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _FakeChat()

    fake_module = types.ModuleType("openai")
    setattr(fake_module, "OpenAI", _FakeOpenAI)  # noqa: B010 — ModuleType has no typed attrs
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return calls


def _record_sleeps(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_client_mod.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _make_client(**overrides) -> OpenAICompatClient:
    return OpenAICompatClient(model="m", api_key="k", base_url="https://x", **overrides)


# ---------- happy path ----------


def test_succeeds_first_attempt_no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client_mod, "_retriable_exceptions", lambda: (_RetriableError,))
    sleeps = _record_sleeps(monkeypatch)
    calls = _install_fake_openai(monkeypatch, behavior=lambda i: "result-content")

    out = _make_client().complete(system="s", user="u")
    assert out == "result-content"
    assert calls["n"] == 1
    assert sleeps == []


# ---------- retry semantics ----------


def test_retries_on_retriable_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client_mod, "_retriable_exceptions", lambda: (_RetriableError,))
    sleeps = _record_sleeps(monkeypatch)

    def behavior(i):
        if i < 2:
            raise _RetriableError("transient")
        return "ok"

    calls = _install_fake_openai(monkeypatch, behavior=behavior)

    out = _make_client(max_retries=3, base_delay=1.0).complete(system="s", user="u")
    assert out == "ok"
    assert calls["n"] == 3
    # Two retries → sleeps after attempt 0 and attempt 1: 1.0, 2.0.
    assert sleeps == [1.0, 2.0]


def test_exhausts_retries_then_raises(monkeypatch):
    monkeypatch.setattr(llm_client_mod, "_retriable_exceptions", lambda: (_RetriableError,))
    sleeps = _record_sleeps(monkeypatch)

    def behavior(i):
        raise _RetriableError(f"call-{i}")

    calls = _install_fake_openai(monkeypatch, behavior=behavior)

    with pytest.raises(RuntimeError, match="LLM call failed after retries") as ei:
        _make_client(max_retries=2, base_delay=1.0).complete(system="s", user="u")
    # 3 total attempts, 2 sleeps in between (1, 2). No sleep after final attempt.
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]
    assert isinstance(ei.value.__cause__, _RetriableError)


def test_backoff_sequence_powers_of_two(monkeypatch):
    monkeypatch.setattr(llm_client_mod, "_retriable_exceptions", lambda: (_RetriableError,))
    sleeps = _record_sleeps(monkeypatch)

    def behavior(i):
        raise _RetriableError("nope")

    _install_fake_openai(monkeypatch, behavior=behavior)
    with pytest.raises(RuntimeError):
        _make_client(max_retries=4, base_delay=0.5).complete(system="s", user="u")
    # base_delay * 2**attempt for attempts 0..3 → 0.5, 1.0, 2.0, 4.0
    assert sleeps == [0.5, 1.0, 2.0, 4.0]


# ---------- non-retriable propagates immediately ----------


def test_non_retriable_propagates_no_sleep(monkeypatch):
    """A bug or auth error propagates on the first attempt."""
    monkeypatch.setattr(llm_client_mod, "_retriable_exceptions", lambda: (_RetriableError,))
    sleeps = _record_sleeps(monkeypatch)

    def behavior(i):
        raise _NonRetriableError("auth")

    calls = _install_fake_openai(monkeypatch, behavior=behavior)
    with pytest.raises(_NonRetriableError):
        _make_client(max_retries=5).complete(system="s", user="u")
    assert calls["n"] == 1
    assert sleeps == []


# ---------- env-failure surfaces clearly ----------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm_client_mod.settings, "llm_api_key", "")
    client = OpenAICompatClient(model="m", base_url="https://x")
    with pytest.raises(RuntimeError, match="AD_LLM_API_KEY not set"):
        client.complete(system="s", user="u")
