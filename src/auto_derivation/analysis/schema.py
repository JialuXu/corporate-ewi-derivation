"""Parquet schema constants for the experiment ledger.

The schema is intentionally flat at the top level (one row per ExperimentRun)
with `pareto_entries` as a nested list-struct. parquet handles this natively.

Schema versioning: bump LEDGER_SCHEMA_VERSION on any breaking change. The
reader checks the version on each file and emits a warning when it sees an
older one — it does NOT fail, since old rows remain queryable for history.
"""
from __future__ import annotations

LEDGER_SCHEMA_VERSION = 1

# Fitness 6-tuple order (must match l3_search.fitness):
#   (IV, KS, stability, monotonicity, complexity_neg, redundancy_neg)
FITNESS_DIM_NAMES = ("iv", "ks", "stability", "monotonicity", "complexity_neg", "redundancy_neg")
