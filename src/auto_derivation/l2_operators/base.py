"""Operator base class + type signature.

An Operator compiles its arguments (already-compiled `pl.Expr`s) into a single
`pl.Expr`. The whole expression tree is therefore turned into one `pl.Expr`
that Polars can evaluate over the full panel in a single pass.

Each Operator declares a `TypeSignature` so the type-checker can validate
trees before evaluation. The signature also lets future GP search filter
candidate primitives by required types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl

from auto_derivation.l1_data.types import LogicalType

WindowKind = str  # "none" | "rolling_n" | "rolling_quarters" | "rolling_days"


@dataclass(frozen=True)
class TypeSignature:
    in_types: tuple[LogicalType, ...]
    out_type: LogicalType
    n_literal_args: int = 0  # int literals (e.g. window length) trailing the field args
    window_kind: WindowKind = "none"


@dataclass(frozen=True)
class EvalContext:
    """Per-evaluation context. Currently only carries the panel's date column
    name so window operators know where time lives. Extend later for industry
    column, peer group key, etc."""

    date_col: str = "observation_date"
    industry_col: str = "industry"


class Operator(ABC):
    name: str
    signature: TypeSignature

    @abstractmethod
    def compile(
        self,
        args: Sequence[pl.Expr],
        literals: Sequence[int | float | str],
        ctx: EvalContext,
    ) -> pl.Expr:
        """Compile a Polars expression for this operator.

        `args` are already-compiled child expressions (length == len(in_types)).
        `literals` are integer/float/string trailing arguments such as window
        lengths (length == n_literal_args).
        """
        ...

    def __repr__(self) -> str:
        return f"<Operator {self.name}>"


@dataclass(frozen=True)
class _OperatorImpl(Operator):
    """Concrete Operator built from a name + signature + compile lambda."""

    name: str
    signature: TypeSignature
    _compile: object = field(repr=False)  # callable

    def compile(
        self,
        args: Sequence[pl.Expr],
        literals: Sequence[int | float | str],
        ctx: EvalContext,
    ) -> pl.Expr:
        return self._compile(args, literals, ctx)  # type: ignore[operator]


def make_operator(
    name: str,
    signature: TypeSignature,
    compile_fn,
) -> Operator:
    return _OperatorImpl(name=name, signature=signature, _compile=compile_fn)
