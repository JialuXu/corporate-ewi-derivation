"""Shared fixtures.

We generate a tiny synthetic panel once per session under a tmp dir so that
the registry / operator / evaluator tests can share it.
"""
from __future__ import annotations

import pytest

from auto_derivation.synthetic import generate


@pytest.fixture(scope="session", autouse=True)
def _pin_example_registry():
    """Pin the whole suite to the committed example inventory.

    Without this, a developer who has the private inventory at
    `data/registry/metrics.csv` would run the suite against different data
    than CI — row-count and source-distribution assertions would flip
    depending on the machine.
    """
    from auto_derivation.config import EXAMPLE_REGISTRY_CSV, settings
    from auto_derivation.l1_data.registry import reset_default_registry

    previous = settings.registry_csv
    settings.registry_csv = EXAMPLE_REGISTRY_CSV
    reset_default_registry()
    yield
    settings.registry_csv = previous
    reset_default_registry()


@pytest.fixture(scope="session")
def synthetic_paths(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    panel, labels = generate(out, n_customers=120, n_months=24, seed=42)
    return panel, labels
