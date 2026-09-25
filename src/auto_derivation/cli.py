"""CLI entry point — `derive`.

Subcommands:
- `derive registry list` — show metric counts by source
- `derive gen-synthetic` — generate the small synthetic panel + labels
- `derive eval --expr ... --label ... [--split oot]` — evaluate one expression
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from auto_derivation._logging import setup_logging
from auto_derivation.config import settings
from auto_derivation.evaluation.metrics import score
from auto_derivation.evaluation.splits import lag_features, time_split
from auto_derivation.expression.evaluator import evaluate
from auto_derivation.expression.tree import parse_sexpr
from auto_derivation.l1_data.panel import join_labels
from auto_derivation.l1_data.registry import default_registry
from auto_derivation.l2_operators.registry import default_operator_registry

app = typer.Typer(add_completion=False, help="对公贷后预警指标自动衍生 - Phase 0")
registry_app = typer.Typer(help="MetricRegistry inspection")
app.add_typer(registry_app, name="registry")
analysis_app = typer.Typer(help="Cross-batch ledger / frontier / meta-report")
app.add_typer(analysis_app, name="analysis")
knowledge_app = typer.Typer(help="Industry knowledge corpus build / query")
app.add_typer(knowledge_app, name="knowledge")


@app.callback()
def _main(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase logging verbosity (-v=INFO, -vv=DEBUG). Default WARNING.",
    ),
) -> None:
    """Top-level callback — configures logging for every subcommand."""
    setup_logging(verbose)


@registry_app.command("list")
def registry_list() -> None:
    """List atomic metrics, grouped by source."""
    reg = default_registry()
    typer.echo(f"Total metrics: {len(reg)}")
    typer.echo("By source:")
    for src, n in sorted(reg.sources().items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {src:8s} {n:4d}")


@app.command("gen-synthetic")
def gen_synthetic(
    customers: int = typer.Option(settings.n_synthetic_customers, "--customers"),
    months: int = typer.Option(settings.n_synthetic_months, "--months"),
    seed: int = typer.Option(settings.random_seed, "--seed"),
) -> None:
    """Generate the synthetic panel + labels under data/synthetic/."""
    from auto_derivation.synthetic import generate

    panel, labels = generate(settings.synthetic_dir, customers, months, seed)
    typer.echo(f"panel  → {panel}")
    typer.echo(f"labels → {labels}")


_EXPR_OPT = typer.Option(..., "--expr", help="S-expression")
_LABEL_OPT = typer.Option("Y_overdue_3m", "--label")
_SPLIT_OPT = typer.Option("oot", "--split", help="train | valid | oot | all")
_PANEL_OPT = typer.Option(None, "--panel", help="Override default synthetic panel path")
_LABELS_OPT = typer.Option(None, "--labels", help="Override default synthetic labels path")
_LAG_OPT = typer.Option(0, "--lag-months", help="Lag features by N months (e.g. 3 for 提前3月预警)")


@app.command("eval")
def eval_expr(
    expr: str = _EXPR_OPT,
    label: str = _LABEL_OPT,
    split: str = _SPLIT_OPT,
    panel_path: Path = _PANEL_OPT,
    labels_path: Path = _LABELS_OPT,
    lag_months: int = _LAG_OPT,
) -> None:
    """Evaluate one S-expression and print IV / KS / Gini / trigger rate."""
    panel = panel_path or settings.panel_path
    labels = labels_path or settings.labels_path
    if not panel.exists():
        raise typer.BadParameter(f"Panel not found: {panel}. Run `derive gen-synthetic` first.")

    ops = default_operator_registry()
    tree = parse_sexpr(expr, op_names=set(ops.names()))

    df, tc = evaluate(tree, panel)
    typer.echo(
        f"[typecheck] out_type={tc.out_type} depth={tc.depth} leaves={tc.leaves}"
    )
    for w in tc.warnings:
        typer.echo(f"[warning] {w}")

    df = join_labels(df, labels, label_name=label)

    if lag_months > 0:
        df = lag_features(df, ["expr_value"], months=lag_months)

    if split != "all":
        train, valid, oot, dates = time_split(df)
        sel = {"train": train, "valid": valid, "oot": oot}[split]
        typer.echo(
            f"[split] {split}: dates ≤ {dates.train_end} | "
            f"valid: {dates.train_end} – {dates.valid_end} | "
            f"oot: {dates.valid_end} – {dates.oot_end}"
        )
        target = sel
    else:
        target = df

    # Drop nulls; drop inf only when the result is numeric (BOOL has no is_finite).
    target = target.filter(pl.col("expr_value").is_not_null())
    if target.schema["expr_value"].is_numeric():
        target = target.filter(pl.col("expr_value").is_finite())

    metrics = score(target, value_col="expr_value", label_col=label)
    typer.echo(f"[score] split={split} n={metrics.n} n_pos={metrics.n_pos}")
    typer.echo(f"  IV   = {metrics.iv:.4f}")
    typer.echo(f"  KS   = {metrics.ks:.4f}")
    typer.echo(f"  Gini = {metrics.gini:.4f}")
    if metrics.trigger_rate is not None:
        typer.echo(f"  trigger_rate = {metrics.trigger_rate:.4f}")


_SEARCH_LABEL_OPT = typer.Option("Y_npl_12m", "--label")
_SEARCH_INDUSTRY_OPT = typer.Option(None, "--industry", help="Restrict search to one industry")
_SEARCH_POP_OPT = typer.Option(60, "--pop", help="Population size")
_SEARCH_GENS_OPT = typer.Option(15, "--gens", help="Number of generations")
_SEARCH_SEED_OPT = typer.Option(0, "--seed")
_SEARCH_TOPK_OPT = typer.Option(8, "--topk", help="Display top-K Pareto-front members")
# Per-run JSON persistence (P1 observability).
_SEARCH_OUT_DIR_OPT = typer.Option(
    None, "--out-dir",
    help="Where to persist the run (defaults to settings.runs_dir = data/runs/).",
)
# Cross-batch parquet ledger (A.1 analysis).
_SEARCH_RECORD_OPT = typer.Option(False, "--record", help="Append run to the cross-batch ledger")
_SEARCH_LEDGER_OPT = typer.Option(
    None, "--ledger-dir", help="Override default ledger directory"
)
_SEARCH_GIT_OPT = typer.Option(None, "--git-sha", help="Tag run with a git commit sha")
_SEARCH_COACH_OPT = typer.Option(
    "mock", "--coach",
    help="LLM coach: 'mock' (no API calls) or 'real' (OpenAI-compatible, "
    "reads AD_LLM_* from .env.local).",
)


@app.command("search")
def search_cmd(
    label: str = _SEARCH_LABEL_OPT,
    industry: str | None = _SEARCH_INDUSTRY_OPT,
    pop: int = _SEARCH_POP_OPT,
    gens: int = _SEARCH_GENS_OPT,
    seed: int = _SEARCH_SEED_OPT,
    topk: int = _SEARCH_TOPK_OPT,
    panel_path: Path = _PANEL_OPT,
    labels_path: Path = _LABELS_OPT,
    out_dir: Path | None = _SEARCH_OUT_DIR_OPT,
    record: bool = _SEARCH_RECORD_OPT,
    ledger_dir: Path | None = _SEARCH_LEDGER_OPT,
    git_sha: str | None = _SEARCH_GIT_OPT,
    coach: str = _SEARCH_COACH_OPT,
) -> None:
    """Run GP search and print the Pareto front."""
    from auto_derivation.l3_search.fitness import INVALID_FITNESS, FitnessConfig
    from auto_derivation.l3_search.llm_coach import LLMCoach
    from auto_derivation.l3_search.search import GPConfig, search
    from auto_derivation.persistence import save_search_run

    panel = panel_path or settings.panel_path
    labels = labels_path or settings.labels_path
    if not panel.exists():
        raise typer.BadParameter(f"Panel not found: {panel}. Run `derive gen-synthetic` first.")

    coach_obj: LLMCoach | None
    if coach == "real":
        from auto_derivation.l3_search.llm_coach_real import OpenAICompatCoach

        coach_obj = OpenAICompatCoach()
    elif coach == "mock":
        coach_obj = None  # search() defaults to MockLLMCoach
    else:
        raise typer.BadParameter(f"--coach must be 'mock' or 'real', got {coach!r}")

    cfg = GPConfig(pop_size=pop, n_gens=gens, seed=seed)
    fc = FitnessConfig(label_col=label, industry=industry)
    res = search(
        panel, labels, label=label, industry=industry,
        config=cfg, coach=coach_obj, fitness_cfg=fc,
    )

    typer.echo(f"=== history (label={label}, industry={industry or 'all'}) ===")
    for h in res.history:
        typer.echo(
            f"  gen {h['gen']:>2}  valid={h['valid']:>3}  "
            f"front0={h['front0_size']:>3}  best_iv={h['best_iv']:.3f}  best_ks={h['best_ks']:.3f}"
        )

    valid = [
        p for p in res.pareto_front
        if p.fitness is not None and p.fitness != INVALID_FITNESS
    ]
    valid.sort(key=lambda p: -(p.fitness or (0,))[0])
    typer.echo(f"\n=== Pareto front (showing top {min(topk, len(valid))} of {len(valid)}) ===")
    for ind in valid[:topk]:
        f = ind.fitness
        assert f is not None  # narrowed by filter above
        typer.echo(
            f"  IV={f[0]:.3f}  KS={f[1]:.3f}  stab={f[2]:.3f}  mono={f[3]:.3f}  "
            f"leaves={int(-f[4])}  redund={(-f[5]):.3f}"
        )
        typer.echo(f"    {ind.expr.to_sexpr()}")

    # P1 observability: always-on per-run JSON persistence.
    run_dir = save_search_run(
        pareto=valid,
        history=res.history,
        label=label,
        scope=industry,
        gp_config=cfg,
        fitness_config=fc,
        panel_path=panel,
        out_dir=out_dir,
    )
    typer.echo(f"\n[persisted] {run_dir}")

    # A.1 cross-batch analysis ledger (opt-in via --record).
    if record:
        from auto_derivation.analysis.ledger import RunLedger, record_search_result

        ledger = RunLedger(ledger_dir or settings.ledger_dir)
        run = record_search_result(
            res, label=label, industry=industry, seed=seed,
            gp_config=cfg, fitness_config=fc,
            panel_path=panel, registry_csv=settings.registry_csv,
            git_sha=git_sha, ledger=ledger,
        )
        typer.echo(
            f"[ledger] recorded run_id={run.run_id} "
            f"({len(run.pareto_entries)} pareto entries) → {ledger.root}"
        )


_EXPLAIN_EXPR_OPT = typer.Option(..., "--expr", help="S-expression to explain")
_EXPLAIN_LABEL_OPT = typer.Option("Y_npl_12m", "--label")
_EXPLAIN_MODEL_OPT = typer.Option(
    None,
    "--model",
    help="LLM model id (defaults to AD_LLM_MODEL from .env / settings)",
)
_EXPLAIN_SAVE_OPT = typer.Option(
    True, "--save/--no-save",
    help="Persist the card under data/cards/<sha12[:2]>/<sha12>.json (default on).",
)


@app.command("explain")
def explain_cmd(
    expr: str = _EXPLAIN_EXPR_OPT,
    label: str = _EXPLAIN_LABEL_OPT,
    model: str | None = _EXPLAIN_MODEL_OPT,
    panel_path: Path = _PANEL_OPT,
    labels_path: Path = _LABELS_OPT,
    save: bool = _EXPLAIN_SAVE_OPT,
) -> None:
    """Generate a business explanation card via the configured LLM provider.

    Reads provider config from .env.local (AD_LLM_API_KEY / _BASE_URL / _MODEL)."""
    from auto_derivation.l4_explain.llm_client import OpenAICompatClient
    from auto_derivation.l4_explain.runner import explain as explain_fn
    from auto_derivation.persistence import save_explanation_card

    panel = panel_path or settings.panel_path
    labels = labels_path or settings.labels_path
    if not panel.exists():
        raise typer.BadParameter(f"Panel not found: {panel}. Run `derive gen-synthetic` first.")

    ops = default_operator_registry()
    tree = parse_sexpr(expr, op_names=set(ops.names()))

    client = OpenAICompatClient(model=model)
    result = explain_fn(
        tree, panel_path=panel, labels_path=labels, label=label, llm_client=client,
    )

    typer.echo(f"指标名: {result.card.metric_name_cn}")
    typer.echo(f"业务解释: {result.card.business_explanation}")
    typer.echo(f"适用场景: {result.card.use_cases}")
    typer.echo(f"差异点:   {result.card.diff_vs_existing}")
    typer.echo(f"阈值建议: {result.card.threshold_advice}")
    typer.echo("复核要点:")
    for c in result.card.review_checklist:
        typer.echo(f"  - {c}")
    if result.consistency_warnings:
        typer.echo("\n[一致性告警]")
        for w in result.consistency_warnings:
            typer.echo(f"  ⚠ {w}")

    if save:
        path = save_explanation_card(expr=tree, label=label, result=result)
        typer.echo(f"\n[persisted] {path}")


_LEDGER_LIST_LABEL_OPT = typer.Option(None, "--label", help="Filter by label")
_LEDGER_LIST_INDUSTRY_OPT = typer.Option(None, "--industry", help="Filter by industry")
_LEDGER_LIST_SINCE_OPT = typer.Option(None, "--since", help="ISO date (YYYY-MM-DD)")
_LEDGER_LIST_DIR_OPT = typer.Option(None, "--ledger-dir", help="Override default ledger directory")


@analysis_app.command("ledger-list")
def analysis_ledger_list(
    label: str | None = _LEDGER_LIST_LABEL_OPT,
    industry: str | None = _LEDGER_LIST_INDUSTRY_OPT,
    since: str | None = _LEDGER_LIST_SINCE_OPT,
    ledger_dir: Path | None = _LEDGER_LIST_DIR_OPT,
) -> None:
    """List recorded GP search runs in the ledger."""
    from datetime import datetime

    from auto_derivation.analysis.ledger import RunLedger

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError as e:
            raise typer.BadParameter(f"--since must be ISO date: {e}") from e

    ledger = RunLedger(ledger_dir or settings.ledger_dir)
    runs = ledger.query(label=label, industry=industry, since=since_dt)
    if not runs:
        typer.echo(f"No runs found under {ledger.root}")
        return
    typer.echo(f"{'run_id':<10} {'timestamp':<20} {'label':<15} {'industry':<10} {'pareto':>6}")
    for r in runs:
        typer.echo(
            f"{r.run_id:<10} {r.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{r.label:<15} {(r.industry or '-'):<10} {len(r.pareto_entries):>6}"
        )


_FRONTIER_INDUSTRY_OPT = typer.Option(None, "--industry", help="Industry slice")
_FRONTIER_OUT_OPT = typer.Option(None, "--out", help="Write Markdown to this path")
_FRONTIER_LEDGER_OPT = typer.Option(None, "--ledger-dir", help="Override ledger directory")


@analysis_app.command("frontier")
def analysis_frontier(
    industry: str | None = _FRONTIER_INDUSTRY_OPT,
    out: Path | None = _FRONTIER_OUT_OPT,
    ledger_dir: Path | None = _FRONTIER_LEDGER_OPT,
) -> None:
    """Print or write the cross-batch global Pareto frontier."""
    from auto_derivation.analysis.frontier import GlobalFrontier, render_frontier
    from auto_derivation.analysis.ledger import RunLedger

    ledger = RunLedger(ledger_dir or settings.ledger_dir)
    frontier = GlobalFrontier(ledger)
    md = render_frontier(frontier, industry)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(md)


_REPORT_CARDS_OPT = typer.Option(
    ..., "--cards-glob",
    help="Glob for ExplanationResult JSON files (e.g. 'explanations/*.json')",
)
_REPORT_SINCE_OPT = typer.Option(None, "--since", help="Filter ledger by ISO date")
_REPORT_INDUSTRY_OPT = typer.Option(None, "--industry", help="Filter by industry")
_REPORT_OUT_OPT = typer.Option(None, "--out", help="Write Markdown to this path")
_REPORT_LEDGER_OPT = typer.Option(None, "--ledger-dir", help="Override ledger dir")
_REPORT_MODEL_OPT = typer.Option(None, "--model", help="LLM model id override")


@analysis_app.command("report")
def analysis_report(
    cards_glob: str = _REPORT_CARDS_OPT,
    industry: str | None = _REPORT_INDUSTRY_OPT,
    since: str | None = _REPORT_SINCE_OPT,
    out: Path | None = _REPORT_OUT_OPT,
    ledger_dir: Path | None = _REPORT_LEDGER_OPT,
    model: str | None = _REPORT_MODEL_OPT,
) -> None:
    """LLM meta-analysis across a batch of explanation cards + ledger runs."""
    import glob
    import json
    from datetime import datetime

    from auto_derivation.analysis.ledger import RunLedger
    from auto_derivation.analysis.meta_report import MetaReportBuilder
    from auto_derivation.analysis.render import render_meta_report
    from auto_derivation.l4_explain.card import ExplanationResult
    from auto_derivation.l4_explain.llm_client import OpenAICompatClient

    card_files = sorted(glob.glob(cards_glob))
    if not card_files:
        raise typer.BadParameter(f"No card files matched: {cards_glob}")
    cards: list[ExplanationResult] = []
    industries_list: list[str | None] = []
    expr_sexprs: list[str | None] = []
    fitness_list: list[list[float] | None] = []
    for cf in card_files:
        with open(cf, encoding="utf-8") as fh:
            doc = json.load(fh)
        # We support two layouts: bare ExplanationResult JSON, or an envelope
        # `{"explanation":..., "industry":..., "expr_sexpr":..., "fitness":...}`.
        if "explanation" in doc and isinstance(doc["explanation"], dict):
            cards.append(ExplanationResult.model_validate(doc["explanation"]))
            industries_list.append(doc.get("industry"))
            expr_sexprs.append(doc.get("expr_sexpr"))
            fitness_list.append(doc.get("fitness"))
        else:
            cards.append(ExplanationResult.model_validate(doc))
            industries_list.append(None)
            expr_sexprs.append(None)
            fitness_list.append(None)

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError as e:
            raise typer.BadParameter(f"--since must be ISO date: {e}") from e
    ledger = RunLedger(ledger_dir or settings.ledger_dir)
    runs = ledger.query(industry=industry, since=since_dt)

    builder = MetaReportBuilder(llm=OpenAICompatClient(model=model))
    report = builder.build(
        cards, expr_sexprs=expr_sexprs, industries=industries_list,
        fitness_list=fitness_list, runs=runs,
    )
    md = render_meta_report(report, runs=runs, cards=cards)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(md)


_SUSPECT_CARDS_OPT = typer.Option(..., "--cards-glob", help="Glob for card JSON files")
_SUSPECT_INDUSTRY_OPT = typer.Option(None, "--industry", help="Industry for RAG filter")
_SUSPECT_LIMIT_OPT = typer.Option(0, "--limit", help="Cap batch size (0 = no limit)")
_SUSPECT_OUT_OPT = typer.Option(None, "--out", help="Write Markdown to this path")
_SUSPECT_MODEL_OPT = typer.Option(None, "--model", help="LLM model id override")


@analysis_app.command("suspect")
def analysis_suspect(
    cards_glob: str = _SUSPECT_CARDS_OPT,
    industry: str | None = _SUSPECT_INDUSTRY_OPT,
    limit: int = _SUSPECT_LIMIT_OPT,
    out: Path | None = _SUSPECT_OUT_OPT,
    model: str | None = _SUSPECT_MODEL_OPT,
) -> None:
    """Score each card with LLM suspicion (industry_fit / drift_risk / spurious_risk)."""
    import glob
    import json

    from auto_derivation.analysis.render import render_suspicion_batch
    from auto_derivation.analysis.suspicion import SuspicionInput, SuspicionScorer
    from auto_derivation.expression.tree import from_dict, parse_sexpr
    from auto_derivation.knowledge.loader import load_corpus
    from auto_derivation.knowledge.retriever import TfidfRetriever
    from auto_derivation.l2_operators.registry import default_operator_registry
    from auto_derivation.l4_explain.card import ExplanationResult
    from auto_derivation.l4_explain.llm_client import OpenAICompatClient

    op_names = set(default_operator_registry().names())
    card_files = sorted(glob.glob(cards_glob))
    if not card_files:
        raise typer.BadParameter(f"No card files matched: {cards_glob}")
    if limit > 0:
        card_files = card_files[:limit]

    inputs: list[SuspicionInput] = []
    cards_only: list[ExplanationResult] = []
    expr_sexprs: list[str] = []
    industries: list[str | None] = []
    for cf in card_files:
        with open(cf, encoding="utf-8") as fh:
            doc = json.load(fh)
        if "explanation" in doc and isinstance(doc["explanation"], dict):
            er = ExplanationResult.model_validate(doc["explanation"])
            ind = doc.get("industry") or industry
            sexpr = doc.get("expr_sexpr")
            expr_dict = doc.get("expr_json_dict")
            if expr_dict:
                expr = from_dict(expr_dict)
            elif sexpr:
                expr = parse_sexpr(sexpr, op_names)
            else:
                raise typer.BadParameter(
                    f"Card {cf} has no expr_sexpr or expr_json_dict; cannot suspect."
                )
            fitness = doc.get("fitness")
        else:
            raise typer.BadParameter(
                f"Card {cf} has no envelope (expected 'explanation' + 'expr_sexpr'); "
                f"add an envelope to provide expr context."
            )
        inputs.append(SuspicionInput(
            explanation=er, expr=expr, industry=ind, fitness=fitness,
        ))
        cards_only.append(er)
        expr_sexprs.append(sexpr or expr.to_sexpr())
        industries.append(ind)

    docs = load_corpus(settings.corpus_dir)
    chunks = [c for d in docs for c in d.chunks]
    retriever = TfidfRetriever(chunks) if chunks else None
    if retriever:
        retriever.fit()

    scorer = SuspicionScorer(
        llm=OpenAICompatClient(model=model),
        retriever=retriever,
    )
    verdicts = scorer.score_batch(inputs)
    md = render_suspicion_batch(verdicts, cards_only, expr_sexprs, industries)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {out}")
    else:
        typer.echo(md)


_KNOWLEDGE_CORPUS_OPT = typer.Option(None, "--corpus-dir", help="Override corpus directory")


@knowledge_app.command("build")
def knowledge_build(corpus_dir: Path | None = _KNOWLEDGE_CORPUS_OPT) -> None:
    """Validate the corpus and report the number of documents / chunks."""
    from auto_derivation.knowledge.loader import load_corpus
    from auto_derivation.knowledge.retriever import TfidfRetriever

    root = corpus_dir or settings.corpus_dir
    docs = load_corpus(root)
    n_chunks = sum(len(d.chunks) for d in docs)
    typer.echo(f"corpus root: {root}")
    typer.echo(f"  {len(docs)} documents, {n_chunks} chunks")
    by_industry: dict[str, int] = {}
    for d in docs:
        key = d.industry or "(cross_cutting)"
        by_industry[key] = by_industry.get(key, 0) + len(d.chunks)
    typer.echo("  chunks by industry:")
    for k in sorted(by_industry):
        typer.echo(f"    {k:<20} {by_industry[k]:>4}")

    # Sanity-fit the retriever to verify the corpus passes through TF-IDF without error.
    chunks = [c for d in docs for c in d.chunks]
    retriever = TfidfRetriever(chunks)
    retriever.fit()
    typer.echo(f"  retriever ready ({len(retriever)} chunks indexed)")


_KQUERY_INDUSTRY_OPT = typer.Option(None, "--industry", help="Filter chunks to this industry")
_KQUERY_TOPK_OPT = typer.Option(5, "--top-k", help="Number of results")


@knowledge_app.command("query")
def knowledge_query(
    query: str = typer.Argument(..., help="Free text query"),
    industry: str | None = _KQUERY_INDUSTRY_OPT,
    top_k: int = _KQUERY_TOPK_OPT,
    corpus_dir: Path | None = _KNOWLEDGE_CORPUS_OPT,
) -> None:
    """Query the corpus with a free-text string."""
    from auto_derivation.knowledge.loader import load_corpus
    from auto_derivation.knowledge.retriever import TfidfRetriever

    root = corpus_dir or settings.corpus_dir
    docs = load_corpus(root)
    chunks = [c for d in docs for c in d.chunks]
    if not chunks:
        typer.echo(f"corpus is empty at {root}; cold-start, no results")
        return
    retriever = TfidfRetriever(chunks)
    retriever.fit()
    hits = retriever.query(query, industry=industry, top_k=top_k)
    if not hits:
        typer.echo("(no matches)")
        return
    for i, (chunk, sim) in enumerate(hits, 1):
        ind = chunk.industry or "-"
        typer.echo(
            f"\n#{i}  score={sim:.3f}  chunk={chunk.chunk_id}  industry={ind}"
        )
        preview = chunk.text[:200] + ("…" if len(chunk.text) > 200 else "")
        typer.echo(f"     {preview}")


if __name__ == "__main__":
    app()
