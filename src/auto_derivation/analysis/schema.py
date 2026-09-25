"""Parquet schema constants for the experiment ledger.

One row per ExperimentRun. Nested fields (gp_config, fitness_config,
pareto_entries, history) are stored as JSON strings, so the schema is flat.

Schema versioning: bump LEDGER_SCHEMA_VERSION on any breaking change. The
reader checks the version on each row, warns when it is newer than
LEDGER_SCHEMA_VERSION, and still loads every row that validates.
"""
from __future__ import annotations

LEDGER_SCHEMA_VERSION = 1

# Fitness 6-tuple order (must match l3_search.fitness):
#   (IV, KS, stability, monotonicity, complexity_neg, redundancy_neg)
FITNESS_DIM_NAMES = ("iv", "ks", "stability", "monotonicity", "complexity_neg", "redundancy_neg")
