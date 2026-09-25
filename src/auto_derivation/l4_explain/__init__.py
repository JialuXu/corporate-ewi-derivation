"""L4 — LLM explanation / naming layer.

Turns an expression tree into a business card:
- 中文指标名 (≤15字)
- 业务解释
- 适用场景
- 与已有指标的差异
- 阈值建议
- 复核要点

Every card goes through a machine consistency check (consistency.py) — fields
mentioned in the explanation must actually appear in the tree, threshold
direction must match the operator's direction, etc.
"""
