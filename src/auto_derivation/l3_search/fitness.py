"""Multi-objective fitness for L3 GP.

Fitness is a 6-vector (all maximized — sign-flip the things we want minimized
so NSGA-II works uniformly):

    (IV, KS_oot, stability, monotonicity, complexity_neg, redundancy_neg)

- IV / KS / monotonicity computed on the chosen evaluation slice (default OOT)
- stability = 1/(1+PSI(train, oot)) → higher is better
- complexity_neg = -(leaves) → smaller trees preferred
- redundancy_neg = -max |Spearman ρ(new, baseline_i)|  → uncorrelated with
  existing metrics is preferred (DESIGN.md §5.2: prevents reinventing wheels)

Caching: fitness is keyed on (expr_json, label, industry). The cache lives on
the FitnessEvaluator instance, so a single search run reuses results across
generations but separate runs start fresh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import polars as pl

from auto_derivation.evaluation.metrics import iv as compute_iv
from auto_derivation.evaluation.metrics import ks as compute_ks
from auto_derivation.evaluation.splits import time_split
from auto_derivation.evaluation.stability import psi
from auto_derivation.expression.evaluator import compile_expr
from auto_derivation.expression.tree import ExprNode, to_canonical_json
from auto_derivation.expression.typecheck import TypeCheckError, typecheck
from auto_derivation.l1_data.registry import MetricRegistry, default_registry
from auto_derivation.l2_operators.base import EvalContext
from auto_derivation.l2_operators.registry import (
    OperatorRegistry,
    default_operator_registry,
)

FITNESS_DIM = 6
INVALID_FITNESS: tuple[float, ...] = (-1.0,) * FITNESS_DIM


def _count_leaves(node: ExprNode) -> int:
    from auto_derivation.expression.tree import FieldNode, LiteralNode, OpNode

    if isinstance(node, (FieldNode, LiteralNode)):
        return 1
    if isinstance(node, OpNode):
        return sum(_count_leaves(c) for c in node.children)
    return 0  # pragma: no cover


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ. Returns 0 if either side is constant (or too few samples)."""
    if len(a) < 5 or len(b) < 5:
        return 0.0
    ra = pl.Series(a).rank().to_numpy()
    rb = pl.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _monotonicity(values: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> float:
    """Monotonicity of label rate vs value-bin index. Returns |Spearman ρ|
    in [0, 1] — higher means score and event-rate move together cleanly."""
    if len(values) == 0:
        return 0.0
    if set(np.unique(values)).issubset({0, 1}):
        # Boolean expressions: just compare event-rate when triggered vs not.
        on = labels[values == 1]
        off = labels[values == 0]
        if len(on) == 0 or len(off) == 0:
            return 0.0
        return float(abs(on.mean() - off.mean()))
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    bins = np.digitize(values, edges[1:-1])
    rates = []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() == 0:
            continue
        rates.append(labels[m].mean())
    if len(rates) < 2:
        return 0.0
    rates_arr = np.asarray(rates)
    bin_idx = np.arange(len(rates_arr), dtype=float)
    return float(abs(_spearman(bin_idx, rates_arr)))


@dataclass
class FitnessConfig:
    label_col: str = "Y_overdue_3m"
    industry: str | None = None  # filter to one industry; None = all
    train_frac: float = 0.6
    valid_frac: float = 0.2
    n_iv_bins: int = 10
    n_psi_bins: int = 10
    redundancy_baseline_metric_ids: tuple[str, ...] = ()
    # When non-empty, redundancy = max |Spearman ρ(score, panel[m])| over m.
    # Empty → redundancy term is 0 (e.g. when running on synthetic data
    # with no curated baseline library yet).


@dataclass
class FitnessEvaluator:
    """Stateful evaluator: holds the joined panel (so we don't re-pivot per
    candidate) and a per-(expr, label, industry) cache of fitness vectors."""

    panel: pl.DataFrame  # wide panel + label column joined
    config: FitnessConfig
    metric_registry: MetricRegistry = field(default_factory=default_registry)
    op_registry: OperatorRegistry = field(default_factory=default_operator_registry)
    _cache: dict[tuple[str, str, str | None], tuple[float, ...]] = field(
        default_factory=dict, init=False
    )

    @cached_property
    def _filtered_panel(self) -> pl.DataFrame:
        df = self.panel
        if self.config.industry:
            df = df.filter(pl.col("industry") == self.config.industry)
        return df

    @cached_property
    def _splits(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        train, _valid, oot, _dates = time_split(
            self._filtered_panel,
            train_frac=self.config.train_frac,
            valid_frac=self.config.valid_frac,
        )
        return train, oot

    def evaluate(self, expr: ExprNode) -> tuple[float, ...]:
        # Canonical form so LiteralNode(0) and LiteralNode(0.0) share a key.
        key = (to_canonical_json(expr), self.config.label_col, self.config.industry)
        if key in self._cache:
            return self._cache[key]

        try:
            typecheck(expr, metric_registry=self.metric_registry, op_registry=self.op_registry)
        except (TypeCheckError, KeyError):
            self._cache[key] = INVALID_FITNESS
            return INVALID_FITNESS

        try:
            ctx = EvalContext()
            compiled = compile_expr(
                expr,
                metrics=self.metric_registry,
                ops=self.op_registry,
                ctx=ctx,
            )
            train_df, oot_df = self._splits
            train_with = train_df.with_columns(compiled.alias("expr_value"))
            oot_with = oot_df.with_columns(compiled.alias("expr_value"))
        except Exception:
            # Compilation or runtime error (e.g. div by zero across types) → invalid.
            self._cache[key] = INVALID_FITNESS
            return INVALID_FITNESS

        # Drop nulls + non-finite in numeric columns.
        train_clean = _clean(train_with, "expr_value", self.config.label_col)
        oot_clean = _clean(oot_with, "expr_value", self.config.label_col)
        n_pos = oot_clean.select(pl.col(self.config.label_col).sum()).item()
        if oot_clean.height < 20 or n_pos < 1:
            self._cache[key] = INVALID_FITNESS
            return INVALID_FITNESS

        v_oot = oot_clean["expr_value"].to_numpy().astype(float)
        y_oot = oot_clean[self.config.label_col].to_numpy().astype(int)

        iv_v = compute_iv(v_oot, y_oot, n_bins=self.config.n_iv_bins)
        ks_v = compute_ks(v_oot, y_oot)
        mono = _monotonicity(v_oot, y_oot)

        if train_clean.height >= 20:
            v_train = train_clean["expr_value"].to_numpy().astype(float)
            psi_v = psi(v_train, v_oot, n_bins=self.config.n_psi_bins)
            stability = 1.0 / (1.0 + max(psi_v, 0.0))
        else:
            stability = 0.0

        complexity_neg = float(-_count_leaves(expr))
        redundancy_neg = -self._redundancy(v_oot, oot_clean)

        result = (iv_v, ks_v, stability, mono, complexity_neg, redundancy_neg)
        self._cache[key] = result
        return result

    def _redundancy(self, score: np.ndarray, panel_slice: pl.DataFrame) -> float:
        if not self.config.redundancy_baseline_metric_ids:
            return 0.0
        max_corr = 0.0
        for mid in self.config.redundancy_baseline_metric_ids:
            if mid not in panel_slice.columns:
                continue
            other = panel_slice[mid].to_numpy()
            mask = np.isfinite(other) & np.isfinite(score)
            if mask.sum() < 5:
                continue
            corr = abs(_spearman(score[mask], other[mask]))
            if corr > max_corr:
                max_corr = corr
        return max_corr


def _clean(df: pl.DataFrame, value_col: str, label_col: str) -> pl.DataFrame:
    out = df.filter(pl.col(value_col).is_not_null() & pl.col(label_col).is_not_null())
    if out.schema[value_col].is_numeric():
        out = out.filter(pl.col(value_col).is_finite())
    return out


