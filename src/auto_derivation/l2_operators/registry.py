"""OperatorRegistry — global name → Operator lookup."""
from __future__ import annotations

from .arith import all_arith
from .base import Operator
from .crosssrc import all_crosssrc
from .event import all_event
from .logic import all_logic
from .peer import all_peer
from .relation import all_relation
from .temporal import all_temporal


class OperatorRegistry:
    def __init__(self, operators: list[Operator]):
        self._by_name: dict[str, Operator] = {}
        for op in operators:
            if op.name in self._by_name:
                raise ValueError(f"Duplicate operator: {op.name}")
            self._by_name[op.name] = op

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def get(self, name: str) -> Operator:
        try:
            return self._by_name[name]
        except KeyError as e:
            raise KeyError(f"Unknown operator: {name}") from e

    def names(self) -> list[str]:
        return list(self._by_name.keys())


_default: OperatorRegistry | None = None


def default_operator_registry() -> OperatorRegistry:
    global _default
    if _default is None:
        _default = OperatorRegistry(
            [
                *all_temporal(),
                *all_event(),
                *all_peer(),
                *all_logic(),
                *all_crosssrc(),
                *all_arith(),
                *all_relation(),
            ]
        )
    return _default
