"""Peer-comparison operators (use industry as the peer group)."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from auto_derivation.l1_data.types import LogicalType

from .base import EvalContext, Operator, TypeSignature, make_operator


def _peer_rank(args: Sequence[pl.Expr], _literals: Sequence, ctx: EvalContext) -> pl.Expr:
    """Percentile rank within (industry, observation_date) cohort, in [0, 1]."""
    (x,) = args
    return x.rank(method="average").over([ctx.industry_col, ctx.date_col]) / pl.len().over(
        [ctx.industry_col, ctx.date_col]
    )


def _peer_deviation(args: Sequence[pl.Expr], _literals: Sequence, ctx: EvalContext) -> pl.Expr:
    """(value − cohort mean) / cohort std."""
    (x,) = args
    grp = [ctx.industry_col, ctx.date_col]
    mean = x.mean().over(grp)
    std = x.std().over(grp)
    return (x - mean) / std


def all_peer() -> list[Operator]:
    return [
        make_operator(
            "PeerRank",
            TypeSignature(
                in_types=(LogicalType.NUMERIC,),
                out_type=LogicalType.RATIO,
                n_literal_args=0,
                window_kind="none",
            ),
            _peer_rank,
        ),
        make_operator(
            "PeerDeviation",
            TypeSignature(
                in_types=(LogicalType.NUMERIC,),
                out_type=LogicalType.NUMERIC,
                n_literal_args=0,
                window_kind="none",
            ),
            _peer_deviation,
        ),
    ]
