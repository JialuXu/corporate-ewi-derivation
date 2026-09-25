"""Stability metrics: PSI, KL divergence, Wasserstein distance.

DESIGN.md §8 pitfall 1: PSI alone is unreliable on small 对公 samples; provide
KL and Wasserstein as cross-checks.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _binned_dist(x: np.ndarray, edges: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    p = counts.astype(float) + eps
    return p / p.sum()


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two samples.

    Bin edges are derived from `expected`'s quantiles; `actual` is bucketed
    into the same edges. Convention: PSI < 0.1 stable, 0.1-0.25 watch, >0.25 unstable.
    """
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    p = _binned_dist(expected, edges)
    q = _binned_dist(actual, edges)
    return float(((p - q) * np.log(p / q)).sum())


def kl_divergence(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """KL(expected || actual) — asymmetric. Same binning convention as PSI."""
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    p = _binned_dist(expected, edges)
    q = _binned_dist(actual, edges)
    return float((p * np.log(p / q)).sum())


def wasserstein(expected: np.ndarray, actual: np.ndarray) -> float:
    """1D Wasserstein-1 distance (a.k.a. Earth-Mover's distance)."""
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    return float(stats.wasserstein_distance(expected, actual))
