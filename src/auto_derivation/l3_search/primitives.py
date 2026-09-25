"""Typed primitive set for GP.

Indexes the L2 operator registry by output type so that a GP node looking for
a sub-expression of type T can pick from the operators / fields that produce
something compatible with T.

Compatibility mirrors `expression.typecheck._types_compatible`:
- exact match always works
- INT and RATIO are accepted wherever NUMERIC is expected
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from auto_derivation.l1_data.registry import MetricMeta, MetricRegistry, default_registry
from auto_derivation.l1_data.types import LogicalType
from auto_derivation.l2_operators.base import Operator
from auto_derivation.l2_operators.registry import (
    OperatorRegistry,
    default_operator_registry,
)

logger = logging.getLogger(__name__)

_NUMERIC_LOGICAL_TYPES: frozenset[LogicalType] = frozenset({
    LogicalType.NUMERIC, LogicalType.INT, LogicalType.RATIO, LogicalType.BOOL,
})


def _is_compatible(actual: LogicalType, expected: LogicalType) -> bool:
    if actual == expected:
        return True
    return expected == LogicalType.NUMERIC and actual in (
        LogicalType.INT,
        LogicalType.RATIO,
    )


# Per-operator literal sampler. Window operators want positive ints; threshold
# operators want a plausible scalar. These ranges are intentionally coarse —
# GP will refine via `literal_jitter` mutation.
LiteralSampler = Callable[[np.random.Generator], int | float | str]

_WINDOW_CHOICES = (1, 3, 6, 12)
_THRESHOLD_CHOICES = (-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 2.0)

LITERAL_SAMPLERS: dict[str, LiteralSampler] = {
    "Delta": lambda r: int(r.choice(_WINDOW_CHOICES)),
    "PctChange": lambda r: int(r.choice(_WINDOW_CHOICES)),
    "Mean": lambda r: int(r.choice(_WINDOW_CHOICES[1:])),  # ≥ 3
    "Std": lambda r: int(r.choice(_WINDOW_CHOICES[1:])),
    "ZScore": lambda r: int(r.choice(_WINDOW_CHOICES[1:])),
    "Slope": lambda r: int(r.choice(_WINDOW_CHOICES[1:])),
    "TsRank": lambda r: int(r.choice(_WINDOW_CHOICES[1:])),
    "Count": lambda r: int(r.choice(_WINDOW_CHOICES[1:])),
    "GT": lambda r: float(r.choice(_THRESHOLD_CHOICES)),
    "LT": lambda r: float(r.choice(_THRESHOLD_CHOICES)),
}


@dataclass
class TypedPrimitiveSet:
    """Operators and fields indexed by output type."""

    op_registry: OperatorRegistry
    metric_registry: MetricRegistry
    excluded_ops: frozenset[str] = frozenset({"AffilSum", "AffilHas"})
    # ↑ Phase 0: relation operators are stubs; exclude from GP until a graph
    # exists. Easy to flip on later by passing excluded_ops=frozenset().
    field_whitelist: frozenset[str] | None = None
    # ↑ When set, only metric_ids in this set are eligible. Use to restrict
    # GP to fields actually present in the panel (avoids wasted evaluations
    # on missing-column errors).
    unsupported_fields: list[MetricMeta] = field(default_factory=list, init=False)
    # ↑ Populated in __post_init__: metrics rejected because the current
    # single-float64 physical schema can't represent CATEGORY/TEXT/DATE.
    # Tracked explicitly so the gap between the L1 type system's contract and
    # the executor's capabilities is visible.

    def __post_init__(self) -> None:
        self._ops_by_out: dict[LogicalType, list[Operator]] = defaultdict(list)
        for op in self.op_registry:
            if op.name in self.excluded_ops:
                continue
            self._ops_by_out[op.signature.out_type].append(op)

        self._fields_by_type: dict[LogicalType, list[MetricMeta]] = defaultdict(list)
        for m in self.metric_registry:
            if self.field_whitelist is not None and m.metric_id not in self.field_whitelist:
                continue
            if m.logical_type in _NUMERIC_LOGICAL_TYPES:
                self._fields_by_type[m.logical_type].append(m)
            else:
                self.unsupported_fields.append(m)
        if self.unsupported_fields:
            counts = Counter(m.logical_type.value for m in self.unsupported_fields)
            logger.info(
                "TypedPrimitiveSet excluded %d metric(s) from GP search: the "
                "physical panel schema is single float64 and cannot carry "
                "CATEGORY/TEXT/DATE. Breakdown by logical_type: %s",
                len(self.unsupported_fields), dict(counts),
            )

    # --- queries ---

    def operators_producing(self, t: LogicalType) -> list[Operator]:
        out: list[Operator] = []
        for produced_t, ops in self._ops_by_out.items():
            if _is_compatible(produced_t, t):
                out.extend(ops)
        return out

    def fields_of_type(self, t: LogicalType) -> list[MetricMeta]:
        out: list[MetricMeta] = []
        for actual_t, fs in self._fields_by_type.items():
            if _is_compatible(actual_t, t):
                out.extend(fs)
        return out

    def operators_with_signature(
        self, in_types: tuple[LogicalType, ...], out_type: LogicalType, n_lit: int
    ) -> list[Operator]:
        """Operators with the *exact* same shape — used for point mutation."""
        return [
            op
            for op in self.op_registry
            if op.name not in self.excluded_ops
            and op.signature.in_types == in_types
            and op.signature.out_type == out_type
            and op.signature.n_literal_args == n_lit
        ]

    def has_field_of_type(self, t: LogicalType) -> bool:
        return len(self.fields_of_type(t)) > 0

    def sample_literal(self, op_name: str, rng: np.random.Generator) -> int | float | str:
        sampler = LITERAL_SAMPLERS.get(op_name)
        if sampler is None:
            # Default: small positive int (covers any future window operator).
            return int(rng.choice(_WINDOW_CHOICES))
        return sampler(rng)


def default_primitive_set() -> TypedPrimitiveSet:
    return TypedPrimitiveSet(
        op_registry=default_operator_registry(),
        metric_registry=default_registry(),
    )
