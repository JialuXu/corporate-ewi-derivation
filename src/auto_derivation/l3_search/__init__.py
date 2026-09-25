"""L3 — GP search engine (gplearn-style typed GP, NSGA-II multi-objective).

The whole layer is built on top of L2's `OperatorRegistry` and L1's
`MetricRegistry`. Phase 0 already enforces that operators carry typed
signatures, so we can treat GP variation as type-respecting tree edits and
let `expression.typecheck` validate every produced tree.
"""
