"""Generate a small synthetic panel + labels for end-to-end testing.

Two customer subgroups:
- group A ("bad"): ~30% of customers; revenue declines + judicial events appear
  in the second year, leading to overdue/downgrade/npl events
- group B ("good"): random walk around stable means, very few events

The synthetic data is deliberately small (~500 customers × 24 months) so that
unit tests can run in seconds, but the injected signal must be strong enough
that PctChange(营业收入) gets IV ≥ 0.2 — that's the regression check in
tests/test_evaluator.py.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from auto_derivation.config import settings
from auto_derivation.l1_data.labels import build_labels
from auto_derivation.l1_data.panel import write_long_panel

# Registered metric_ids we populate. All exist in data/registry/metrics.example.csv
# (guarded by tests/test_registry.py).
METRICS = {
    # 财务 — 季度更新
    "FIN_0001": ("amount", "营业收入"),
    "FIN_0004": ("amount", "净利润"),
    "FIN_0012": ("ratio", "资产负债率"),
    "FIN_0017": ("ratio_pct", "营业收入同比增长率"),
    # 行内信贷 — 日频，月末快照
    "LOAN_0003": ("amount", "合同总金额"),
    "LOAN_0006": ("amount", "贷款余额"),
    # 征信 — 日频
    "CRD_0001": ("amount", "未结清不良类表内信贷余额"),
    "CRD_0007": ("count", "近期征信查询次数"),
    # 司法 — 事件型，按月聚合
    "JUD_0001": ("count", "近一年被执行案件数量"),
    # 工商 — 静态
    "BIZ_0001": ("amount", "注册资本"),
}

INDUSTRIES = ["制造", "批零", "建筑"]


def _month_ends(start: date, n_months: int) -> list[date]:
    """Return n_months consecutive month-end dates starting from `start`'s month."""
    out: list[date] = []
    y, m = start.year, start.month
    for _ in range(n_months):
        next_first = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        out.append(next_first - timedelta(days=1))
        y, m = next_first.year, next_first.month
    return out


def _gen_customer_series(
    rng: np.random.Generator,
    is_bad: bool,
    months: list[date],
    industry: str,
) -> dict[str, np.ndarray]:
    """Per-customer per-month series for each populated metric_id."""
    n = len(months)
    out: dict[str, np.ndarray] = {}

    industry_scale = {"制造": 1.2, "批零": 0.8, "建筑": 1.0}[industry]

    # Revenue: stable for good; for bad, declines starting month 12.
    base_rev = rng.normal(loc=10_000_000, scale=1_500_000) * industry_scale
    rev = np.full(n, base_rev) + rng.normal(scale=base_rev * 0.05, size=n)
    if is_bad:
        decay = np.where(np.arange(n) < 12, 1.0, 1.0 - 0.04 * (np.arange(n) - 12))
        rev = rev * np.clip(decay, 0.4, 1.0)
    out["FIN_0001"] = rev

    # Net profit ~ revenue * margin, bad has shrinking margin.
    margin = rng.normal(loc=0.1, scale=0.02, size=n)
    if is_bad:
        margin = margin - 0.005 * np.maximum(0, np.arange(n) - 12)
    out["FIN_0004"] = rev * margin

    # Liability ratio drifts up for bad customers.
    lev = rng.normal(loc=0.55, scale=0.05, size=n)
    if is_bad:
        lev = lev + 0.01 * np.maximum(0, np.arange(n) - 12)
    out["FIN_0012"] = np.clip(lev, 0.1, 0.99)

    # Revenue YoY growth (computed numerically for honesty).
    yoy = np.full(n, np.nan)
    for i in range(12, n):
        yoy[i] = (rev[i] - rev[i - 12]) / rev[i - 12]
    out["FIN_0017"] = yoy

    # Loan amount + outstanding balance.
    contract = rng.normal(loc=5_000_000, scale=1_000_000) * industry_scale
    out["LOAN_0003"] = np.full(n, contract)
    bal = np.cumsum(rng.normal(loc=0, scale=50_000, size=n)) + contract * 0.7
    out["LOAN_0006"] = np.clip(bal, contract * 0.1, contract)

    # Credit-bureau bad-class balance (rare for good, growing for bad).
    npl_bal = np.zeros(n)
    if is_bad:
        npl_bal = np.where(
            np.arange(n) < 15,
            0.0,
            (np.arange(n) - 14) * 80_000 + rng.normal(scale=20_000, size=n),
        )
    out["CRD_0001"] = np.clip(npl_bal, 0, None)

    # Recent credit-bureau queries (Poisson). Bad customers query more.
    lam_good, lam_bad = 1.0, 3.5
    out["CRD_0007"] = rng.poisson(lam_bad if is_bad else lam_good, size=n).astype(float)

    # Judicial cases (mostly 0; non-zero spike in months 14-18 for bad).
    jud = np.zeros(n)
    if is_bad:
        jud[14:19] = rng.poisson(2.0, size=5).astype(float)
    out["JUD_0001"] = jud

    # Registered capital: static throughout.
    out["BIZ_0001"] = np.full(n, rng.normal(loc=20_000_000, scale=5_000_000))

    return out


def _gen_events(
    rng: np.random.Generator,
    customer_id: str,
    is_bad: bool,
    months: list[date],
) -> list[tuple[str, date, str]]:
    """(entity_id, event_date, event_type) tuples."""
    if not is_bad:
        # ~5% of good customers have a single overdue30 event somewhere.
        if rng.random() < 0.05:
            d = months[rng.integers(0, len(months))]
            return [(customer_id, d, "overdue30")]
        return []
    # Bad: cascade — judicial first, then credit_adverse, then overdue, then downgrade/npl.
    out: list[tuple[str, date, str]] = []
    out.append((customer_id, months[14], "judicial_adverse"))
    out.append((customer_id, months[15], "credit_adverse"))
    out.append((customer_id, months[17], "overdue30"))
    if rng.random() < 0.7:
        out.append((customer_id, months[20], "downgrade"))
    if rng.random() < 0.4:
        out.append((customer_id, months[22], "npl"))
    return out


def generate(
    out_dir: Path,
    n_customers: int,
    n_months: int,
    seed: int,
) -> tuple[Path, Path]:
    if n_months < 23:
        raise ValueError(
            f"n_months must be ≥ 23 (got {n_months}): the injected bad-customer "
            "cascade (revenue decline → judicial spike → overdue → npl) spans "
            "months 14–22 and the windows are not adaptive."
        )
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    months = _month_ends(date(2024, 1, 1), n_months)
    n_bad = int(n_customers * 0.3)
    is_bad = np.array([True] * n_bad + [False] * (n_customers - n_bad))
    rng.shuffle(is_bad)

    long_rows: list[dict] = []
    event_rows: list[dict] = []
    for i in range(n_customers):
        cid = f"CUST{i:05d}"
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        bad = bool(is_bad[i])
        series = _gen_customer_series(rng, bad, months, industry)
        for j, d in enumerate(months):
            for metric_id, values in series.items():
                v = values[j]
                long_rows.append(
                    {
                        "entity_id": cid,
                        "observation_date": d,
                        "industry": industry,
                        "metric_id": metric_id,
                        "value": float(v) if v == v else None,  # NaN → NULL
                    }
                )
        for ev in _gen_events(rng, cid, bad, months):
            event_rows.append(
                {"entity_id": ev[0], "event_date": ev[1], "event_type": ev[2]}
            )

    panel_df = pl.DataFrame(long_rows).with_columns(
        pl.col("observation_date").cast(pl.Date)
    )
    panel_path = out_dir / "panel.parquet"
    write_long_panel(panel_df, panel_path)

    events_df = pl.DataFrame(event_rows).with_columns(
        pl.col("event_date").cast(pl.Date)
    )
    obs_dates = pl.Series("observation_date", months).cast(pl.Date)
    labels_df = build_labels(events_df, obs_dates)
    labels_path = out_dir / "labels.parquet"
    labels_df.write_parquet(labels_path)

    return panel_path, labels_path


if __name__ == "__main__":
    p, lbl = generate(
        settings.synthetic_dir,
        settings.n_synthetic_customers,
        settings.n_synthetic_months,
        settings.random_seed,
    )
    print(f"panel  → {p}")
    print(f"labels → {lbl}")
