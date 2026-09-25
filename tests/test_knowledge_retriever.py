"""Tests for the TF-IDF retriever."""
from __future__ import annotations

from pathlib import Path

from auto_derivation.knowledge.corpus import CorpusChunk
from auto_derivation.knowledge.loader import load_corpus
from auto_derivation.knowledge.retriever import TfidfRetriever


def _chunk(text: str, industry: str | None = None, chunk_id: str = "x#x") -> CorpusChunk:
    return CorpusChunk(chunk_id=chunk_id, text=text, industry=industry, headings=[])


def test_empty_corpus_returns_empty_results():
    r = TfidfRetriever([])
    r.fit()
    assert r.query("anything") == []


def test_unfit_retriever_returns_empty():
    r = TfidfRetriever([_chunk("一些文本")])
    # 不调用 fit()
    assert r.query("文本") == []


def test_query_finds_relevant_chunk():
    chunks = [
        _chunk("应收账款占营收 30% 是制造业的正常水位。", "制造", "manu#ar"),
        _chunk("库存周转天数应控制在 60 天以内。", "批零", "rtai#inv"),
        _chunk("工程款回收周期长是建筑业特征。", "建筑", "cons#cycle"),
    ]
    r = TfidfRetriever(chunks)
    r.fit()
    hits = r.query("应收账款", top_k=3)
    assert len(hits) >= 1
    top_chunk, top_score = hits[0]
    assert "应收账款" in top_chunk.text
    assert top_score > 0


def test_industry_filter_excludes_other_industries():
    chunks = [
        _chunk("应收账款占营收 30% 是制造业的正常水位。", "制造", "manu#ar"),
        _chunk("应收账款风险在批零业较小，因 C 端即时结算。", "批零", "rtai#ar"),
        _chunk("通用：异常资金流是跨行业风险。", None, "cross#x"),
    ]
    r = TfidfRetriever(chunks)
    r.fit()
    hits = r.query("应收账款", industry="制造", top_k=3)
    industries = {c.industry for c, _ in hits}
    # 制造或 None 通过；批零应该被过滤掉
    assert "批零" not in industries


def test_cross_cutting_chunks_visible_to_all_industries():
    chunks = [
        _chunk("通用：异常资金流是跨行业风险。", None, "cross#x"),
        _chunk("制造业的特殊存货风险。", "制造", "manu#inv"),
    ]
    r = TfidfRetriever(chunks)
    r.fit()
    hits = r.query("跨行业 风险", industry="建筑", top_k=2)
    # 制造的 chunk 应被过滤，但 cross_cutting 应可见
    industries = {c.industry for c, _ in hits}
    assert None in industries
    assert "制造" not in industries


def test_empty_query_returns_empty():
    chunks = [_chunk("一些内容")]
    r = TfidfRetriever(chunks)
    r.fit()
    assert r.query("") == []
    assert r.query("   ") == []


def test_end_to_end_with_mini_corpus():
    """Load fixtures/mini_corpus and check a realistic query."""
    root = Path(__file__).parent / "fixtures" / "mini_corpus"
    docs = load_corpus(root)
    chunks = [c for d in docs for c in d.chunks]
    r = TfidfRetriever(chunks)
    r.fit()

    hits = r.query("应付账款 杠杆", industry="批零", top_k=3)
    assert len(hits) >= 1
    top = hits[0][0]
    assert "应付账款" in top.text
    # 批零 industry 命中了批零 chunk
    assert top.industry == "批零"
