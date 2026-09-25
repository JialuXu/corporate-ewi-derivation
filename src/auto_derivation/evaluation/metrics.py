"""Discrimination metrics: IV, KS, Gini.

IV uses equal-frequency binning via `np.quantile`.

KS uses scipy.stats.ks_2samp on the score distributions of label==0 vs label==1
(the standard credit-risk KS).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats


@dataclass(frozen=True)
class DiscriminationMetrics:
    iv: float
    ks: float
    gini: float
    n: int
    n_pos: int
    trigger_rate: float | None  # only meaningful for boolean expressions


def _to_arrays(df: pl.DataFrame, value_col: str, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    out = df.drop_nulls([value_col, label_col]).select([value_col, label_col])
    v = out[value_col].to_numpy()
    y = out[label_col].to_numpy().astype(int)
    return v, y


def iv(values: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Information Value with equal-frequency binning + Laplace smoothing."""
    if len(values) == 0:
        return 0.0
    is_bool = set(np.unique(values)).issubset({0, 1})
    if is_bool:
        bins = values.astype(int)
    else:
        try:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
            bins = np.digitize(values, edges[1:-1])
        except Exception:
            return 0.0

    pos_total = labels.sum() + 1e-9
    neg_total = (1 - labels).sum() + 1e-9

    iv_total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        pos = labels[mask].sum()
        neg = (1 - labels[mask]).sum()
        # Laplace smoothing to avoid log(0).
        p = (pos + 0.5) / pos_total
        n = (neg + 0.5) / neg_total
        iv_total += (p - n) * np.log(p / n)
    return float(iv_total)


def ks(values: np.ndarray, labels: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(stats.ks_2samp(pos, neg).statistic)  # type: ignore[attr-defined]


def gini(values: np.ndarray, labels: np.ndarray) -> float:
    """Gini = 2*AUC - 1, computed via Mann-Whitney U."""
    if len(values) == 0:
        return 0.0
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    u, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    auc = u / (len(pos) * len(neg))
    return float(2 * auc - 1)


def score(
    df: pl.DataFrame,
    *,
    value_col: str,
    label_col: str,
    n_bins: int = 10,
) -> DiscriminationMetrics:
    """Single entry-point used by CLI / Phase 2 GP fitness."""
    v, y = _to_arrays(df, value_col, label_col)
    is_bool_expr = set(np.unique(v)).issubset({0, 1}) if len(v) else False
    trigger = float(v.mean()) if is_bool_expr and len(v) else None
    return DiscriminationMetrics(
        iv=iv(v, y, n_bins=n_bins),
        ks=ks(v, y),
        gini=gini(v, y),
        n=len(v),
        n_pos=int(y.sum()),
        trigger_rate=trigger,
    )
