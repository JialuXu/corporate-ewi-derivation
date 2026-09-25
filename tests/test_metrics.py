"""Verify our IV / KS / Gini implementations against scipy / sklearn baselines."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from auto_derivation.evaluation.metrics import gini, iv, ks
from auto_derivation.evaluation.stability import psi, wasserstein


@pytest.fixture
def discrim_data():
    rng = np.random.default_rng(0)
    # values predictive of label
    y = rng.integers(0, 2, size=1000)
    v = rng.normal(loc=y * 0.8, scale=1.0)
    return v, y


def test_ks_matches_scipy(discrim_data):
    v, y = discrim_data
    ours = ks(v, y)
    # getattr: pyright's scipy stubs type the result fields as a placeholder.
    theirs = getattr(stats.ks_2samp(v[y == 1], v[y == 0]), "statistic")  # noqa: B009
    assert ours == pytest.approx(float(theirs))


def test_gini_matches_2auc_minus_1(discrim_data):
    v, y = discrim_data
    ours = gini(v, y)
    theirs = 2 * roc_auc_score(y, v) - 1
    assert ours == pytest.approx(theirs, abs=1e-3)


def test_iv_positive_when_predictive(discrim_data):
    v, y = discrim_data
    assert iv(v, y) > 0.1


def test_iv_near_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=2000)
    v = rng.normal(size=2000)
    assert abs(iv(v, y)) < 0.05


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    a = rng.normal(size=2000)
    b = rng.normal(size=2000)
    val = psi(a, b)
    assert val < 0.05


def test_psi_large_for_shifted_distributions():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    b = rng.normal(loc=2.0, size=2000)
    assert psi(a, b) > 0.5


def test_wasserstein_zero_for_identical():
    rng = np.random.default_rng(4)
    a = rng.normal(size=500)
    b = a.copy()
    assert wasserstein(a, b) == pytest.approx(0.0)
