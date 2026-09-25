"""GP individual — an expression tree + cached fitness vector.

Identity / cache keys use `tree.to_canonical_json` so trees differing only
in literal representation (0 vs 0.0) collapse to the same key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from auto_derivation.expression.tree import ExprNode, to_canonical_json


@dataclass
class Individual:
    expr: ExprNode
    label: str
    industry: str | None = None
    fitness: tuple[float, ...] | None = None
    # NSGA-II bookkeeping (assigned by nsga2.assign_ranks):
    rank: int = -1
    crowding: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expr_key(self) -> str:
        return to_canonical_json(self.expr)

    def __hash__(self) -> int:
        return hash((self.expr_key, self.label, self.industry))
