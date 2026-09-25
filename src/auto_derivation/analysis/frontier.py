"""Global cross-batch Pareto frontier.

Reads all `ExperimentRun`s from a `RunLedger`, groups by industry, then
computes the union-then-non-dominated set per industry. Lets callers ask:

- "What's the current best frontier I've ever seen for 制造?"
- "Did this new run actually expand the frontier, or just dominate itself?"
- "Are runs in this aggregate comparable (same panel signature)?"

We re-use `l3_search.nsga2.dominates` so the dominance relation is
identical to what NSGA-II uses inside a single search.
"""
from __future__ import annotations

from dataclasses import dataclass

from auto_derivation.l3_search.nsga2 import dominates

from .ledger import ExperimentRun, ParetoEntry, RunLedger
from .schema import FITNESS_DIM_NAMES


@dataclass
class ExpansionDelta:
    """How a single new run interacts with the existing global frontier.

    - `new_on_front`: entries from the new run that survive on the merged frontier
    - `existing_dominated`: how many old frontier entries the new run kicked out
    - `net_added`: new_on_front - existing_dominated (informational; can be negative
      if the new run only matches existing without strictly expanding)
    - `panel_signature_consistent`: did this run share panel_signature with the
      majority of prior runs in the same industry? If False, comparing fronts
      across runs is apples-to-oranges and the caller should be careful.
    """

    new_on_front: int
    existing_dominated: int
    net_added: int
    panel_signature_consistent: bool


def _non_dominated(entries: list[ParetoEntry]) -> list[ParetoEntry]:
    """Filter `entries` to those not dominated by any other entry. O(N²)."""
    out: list[ParetoEntry] = []
    for i, e in enumerate(entries):
        ei = tuple(e.fitness)
        beaten = False
        for j, f in enumerate(entries):
            if i == j:
                continue
            fj = tuple(f.fitness)
            # Pad shorter tuples with -inf so unequal fitness dims still compare
            # — by construction they should match, but be defensive.
            if len(ei) != len(fj):
                continue
            if dominates(fj, ei):
                beaten = True
                break
        if not beaten:
            out.append(e)
    return out


class GlobalFrontier:
    """Build cross-batch frontiers by industry from a RunLedger."""

    def __init__(self, ledger: RunLedger):
        self.ledger = ledger
        self._cache: dict[str | None, list[ParetoEntry]] = {}
        self._signatures: dict[str | None, list[str]] = {}

    def _gather(self, industry: str | None) -> tuple[list[ParetoEntry], list[str]]:
        runs = self.ledger.query(industry=industry)
        entries: list[ParetoEntry] = []
        sigs: list[str] = []
        for r in runs:
            entries.extend(r.pareto_entries)
            if r.panel_signature:
                sigs.append(r.panel_signature)
        return entries, sigs

    def front_for(self, industry: str | None) -> list[ParetoEntry]:
        """Return the union-then-non-dominated frontier for one industry slice."""
        if industry in self._cache:
            return self._cache[industry]
        entries, sigs = self._gather(industry)
        front = _non_dominated(entries)
        self._cache[industry] = front
        self._signatures[industry] = sigs
        return front

    def is_dominated(self, fit: list[float], industry: str | None) -> bool:
        """Is `fit` dominated by anything on the current global frontier?"""
        ft = tuple(fit)
        for e in self.front_for(industry):
            if len(e.fitness) != len(ft):
                continue
            if dominates(tuple(e.fitness), ft):
                return True
        return False

    def expansion_delta(self, run: ExperimentRun) -> ExpansionDelta:
        """How does `run` interact with the existing frontier for its industry?

        IMPORTANT: this expects `run` to ALREADY be in the ledger (i.e. you
        called `ledger.append(run)` first). We re-read the full set, identify
        which post-merge frontier entries come from this run vs prior runs,
        and count.
        """
        industry = run.industry
        all_entries, sigs = self._gather(industry)

        # Tag each entry with whether it belongs to this run. We use
        # (expr_json, fitness) as the identity key; expr_json alone could
        # collide across runs if two searches found the same tree.
        run_keys = {(e.expr_json, tuple(e.fitness)) for e in run.pareto_entries}
        front = _non_dominated(all_entries)

        new_on_front = 0
        for e in front:
            if (e.expr_json, tuple(e.fitness)) in run_keys:
                new_on_front += 1

        # Count old front entries that were knocked out. We compute what the
        # frontier WOULD have been without this run, then diff.
        prior = [e for e in all_entries if (e.expr_json, tuple(e.fitness)) not in run_keys]
        prior_front = _non_dominated(prior)
        prior_set = {(e.expr_json, tuple(e.fitness)) for e in prior_front}
        front_set = {(e.expr_json, tuple(e.fitness)) for e in front}
        existing_dominated = len(prior_set - front_set)

        # Signature consistency: does this run's panel_signature match the
        # majority of prior signatures in this industry?
        prior_sigs = [s for s in sigs if s != run.panel_signature] if run.panel_signature else sigs
        consistent = True
        if prior_sigs and run.panel_signature:
            # majority match
            from collections import Counter

            counts = Counter(prior_sigs)
            modal, _ = counts.most_common(1)[0]
            consistent = modal == run.panel_signature

        return ExpansionDelta(
            new_on_front=new_on_front,
            existing_dominated=existing_dominated,
            net_added=new_on_front - existing_dominated,
            panel_signature_consistent=consistent,
        )


def render_frontier(frontier: GlobalFrontier, industry: str | None) -> str:
    """Markdown table of the current global frontier for one industry."""
    front = frontier.front_for(industry)
    label = industry if industry else "（全部 / 不分行业）"
    if not front:
        return f"# 全局 Pareto 前沿 — {label}\n\n（当前没有已记录的非支配候选）\n"

    # sort by IV desc for readability
    front_sorted = sorted(front, key=lambda e: -e.fitness[0] if e.fitness else 0.0)

    lines = [
        f"# 全局 Pareto 前沿 — {label}",
        "",
        f"共 {len(front_sorted)} 个非支配候选（跨批次合并 + 重新非支配筛选）",
        "",
        "| # | IV | KS | stability | monotonicity | leaves | redundancy | 表达式 |",
        "|---|----|----|-----------|--------------|--------|------------|--------|",
    ]
    for i, e in enumerate(front_sorted, 1):
        f = e.fitness
        if len(f) >= 6:
            iv, ks, stab, mono, comp_neg, redund_neg = f[:6]
            leaves = int(-comp_neg) if comp_neg <= 0 else 0
            redund = -redund_neg if redund_neg <= 0 else 0.0
            lines.append(
                f"| {i} | {iv:.3f} | {ks:.3f} | {stab:.3f} | {mono:.3f} | "
                f"{leaves} | {redund:.3f} | `{e.expr_sexpr}` |"
            )
        else:
            # Future-proofing: dimensions may have grown — show what we have.
            fit_str = ", ".join(
                f"{name}={v:.3f}"
                for name, v in zip(FITNESS_DIM_NAMES, f, strict=False)
            )
            lines.append(f"| {i} | {fit_str} | — | — | — | — | — | `{e.expr_sexpr}` |")
    lines.append("")
    return "\n".join(lines)
