"""LLM coach interface (DESIGN.md §5.3 — three roles).

The coach is a *Protocol*: a real OpenAI-compatible adapter
(`llm_coach_real.py`, reusing the L4 transport in
`auto_derivation.l4_explain.llm_client`) and a no-op mock both implement it.

The coach is optional. Pass `MockLLMCoach()` (or omit) to skip LLM calls
entirely — useful for tests / no-API runs.
"""
from __future__ import annotations

from typing import Protocol

from auto_derivation.expression.tree import ExprNode


class LLMCoach(Protocol):
    def seeds(
        self,
        *,
        label: str,
        industry: str | None,
        available_field_names: list[str],
        available_op_names: list[str],
        k: int = 10,
    ) -> list[ExprNode]:
        """Generate up to `k` 'prior-knowledge' candidate trees. May return
        fewer (or zero). Trees that fail typecheck are discarded by the caller."""
        ...

    def critique(
        self,
        *,
        candidates: list[ExprNode],
    ) -> list[float]:
        """Return one [0, 1] score per candidate — interpretation is "business
        plausibility". The search loop multiplies fitness IV by `(1 + α·score)`
        as a soft hint."""
        ...

    def cross_source_proposals(
        self,
        *,
        coverage_summary: dict[str, int],
        available_field_names: list[str],
        available_op_names: list[str],
        k: int = 5,
    ) -> list[ExprNode]:
        """Suggest under-explored cross-source combinations.
        `coverage_summary` is e.g. `{"FIN": 12, "JUD": 0, ...}` counting
        how many top candidates reference each source."""
        ...


class MockLLMCoach:
    """No-op coach used by tests / no-API mode."""

    def seeds(
        self, *, label, industry, available_field_names, available_op_names, k=10
    ) -> list[ExprNode]:
        return []

    def critique(self, *, candidates) -> list[float]:
        return [0.5] * len(candidates)  # neutral

    def cross_source_proposals(
        self, *, coverage_summary, available_field_names, available_op_names, k=5
    ) -> list[ExprNode]:
        return []
