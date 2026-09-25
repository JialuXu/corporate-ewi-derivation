"""LLM client for L4 explanation generation.

Two implementations:
- OpenAICompatClient: real API; works with any OpenAI-compatible endpoint
  (OpenAI, DeepSeek, Moonshot, vLLM, …). Reads `AD_LLM_API_KEY` /
  `AD_LLM_BASE_URL` / `AD_LLM_MODEL` from settings.
- MockClient: returns canned JSON, used by tests / CI / no-API runs.

Both implement the same `complete(system, user) -> str` interface so the
explanation runner doesn't care which is wired up.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from auto_derivation.config import settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> str: ...


@dataclass
class MockClient:
    """Returns the same canned response every call. Used by tests."""

    canned_json: str
    last_call: dict | None = None

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        self.last_call = {"system": system, "user": user, "max_tokens": max_tokens}
        return self.canned_json


def _retriable_exceptions() -> tuple[type[BaseException], ...]:
    """OpenAI SDK exceptions worth retrying. Lazy-imported because the SDK
    is an optional install at this layer (Mock paths must work without it).

    We retry only on transient failure modes:
      - RateLimitError  : provider asked us to back off
      - APITimeoutError : the request took too long
      - APIConnectionError : transport-level hiccup
    """
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except ImportError:
        return ()
    return (RateLimitError, APITimeoutError, APIConnectionError)


@dataclass
class OpenAICompatClient:
    """Talks to any OpenAI-compatible chat completions API.

    Defaults pulled from `Settings` so configuration lives in `.env.local`.
    Per-instance overrides win if you pass `model=` / `base_url=` / `api_key=`.

    Retry policy: exponential backoff with `base_delay * 2**attempt` seconds
    between attempts. Only the SDK's transient exception types are retried, so
    auth / schema bugs surface on the first attempt.
    """

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_retries: int = 2
    base_delay: float = 1.0
    temperature: float = 0.2

    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. Run `uv sync` to install."
            ) from e

        key = self.api_key or settings.llm_api_key
        if not key:
            raise RuntimeError(
                "AD_LLM_API_KEY not set. Add it to .env.local or export it."
            )
        base_url = self.base_url or settings.llm_base_url
        model = self.model or settings.llm_model

        client = OpenAI(api_key=key, base_url=base_url)
        retriable = _retriable_exceptions()
        last_err: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    "llm.complete model=%s attempt=%d/%d max_tokens=%d",
                    model, attempt + 1, self.max_retries + 1, max_tokens,
                )
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or ""
                return content
            except retriable as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                delay = self.base_delay * (2 ** attempt)
                logger.warning(
                    "llm.complete retriable error (%s); sleeping %.2fs before retry",
                    type(e).__name__, delay,
                )
                time.sleep(delay)
        assert last_err is not None
        raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def parse_json_response(raw: str) -> dict:
    """Tolerate common LLM JSON glitches: code fences, trailing prose."""
    s = raw.strip()
    if s.startswith("```"):
        # strip first fence line + last fence
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Find the first '{' and matching final '}' — handles trailing prose.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {raw[:200]!r}")
    return json.loads(s[start : end + 1])
