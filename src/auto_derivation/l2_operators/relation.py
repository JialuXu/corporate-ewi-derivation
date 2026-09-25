"""Relation/contagion operators — Phase 0 stubs.

These require an entity-relation graph (associated parties, guarantors, supply
chain) which Phase 0 does not model. The signatures are declared so future
trees can reference them; calling .compile raises NotImplementedError.
"""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import Operator, TypeSignature, make_operator


def _not_implemented(*_args, **_kw) -> pl.Expr:
    raise NotImplementedError(
        "Relation operators require an entity-graph layer (Phase 1+). "
        "The signature is registered so trees can be constructed and shown, "
        "but evaluation requires real data."
    )


def all_relation() -> list[Operator]:
    return [
        make_operator(
            "AffilSum",
            TypeSignature(
                in_types=(LogicalType.NUMERIC,),
                out_type=LogicalType.NUMERIC,
            ),
            _not_implemented,
        ),
        make_operator(
            "AffilHas",
            TypeSignature(
                in_types=(LogicalType.BOOL,),
                out_type=LogicalType.BOOL,
            ),
            _not_implemented,
        ),
    ]


def _unused(_: Sequence[pl.Expr]) -> None:  # keep `Sequence` import stable
    return None
