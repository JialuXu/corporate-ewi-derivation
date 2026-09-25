"""Tests for the corpus loader and markdown chunker."""
from __future__ import annotations

from pathlib import Path

from auto_derivation.knowledge.loader import (
    _strip_markdown,
    chunk_markdown,
    load_corpus,
)


def test_strip_markdown_removes_syntax():
    raw = "# 标题\n\n这是 **加粗** 和 *斜体*，还有 `代码`，最后 [链接](http://x)。"
    plain = _strip_markdown(raw)
    assert "标题" in plain
    assert "加粗" in plain
    assert "斜体" in plain
    assert "代码" in plain
    assert "链接" in plain
    assert "**" not in plain
    assert "[" not in plain
    assert "(" not in plain


def test_chunk_markdown_splits_by_heading():
    raw = """# 主标题

前置段落。

## 子节A

子节A 的内容是这一段。

## 子节B

子节B 的内容是另一段。

### 三级标题

三级内容。
"""
    chunks = chunk_markdown(raw)
    heading_paths = [h for h, _ in chunks]
    titles = [path[-1] if path else "" for path in heading_paths]
    assert "主标题" in titles
    assert "子节A" in titles
    assert "子节B" in titles
    assert "三级标题" in titles
    # chunk text must include the heading text itself
    body_a = next(b for h, b in chunks if h and h[-1] == "子节A")
    assert "子节A" in body_a
    assert "子节A 的内容" in body_a


def test_chunk_markdown_empty_text():
    assert chunk_markdown("") == []


def test_chunk_markdown_no_headings_becomes_one_chunk():
    raw = "纯文本，没有任何 heading。"
    chunks = chunk_markdown(raw)
    assert len(chunks) == 1
    headings, body = chunks[0]
    assert headings == []
    assert "纯文本" in body


def test_load_corpus_mini_fixture():
    root = Path(__file__).parent / "fixtures" / "mini_corpus"
    docs = load_corpus(root)
    # Two industry docs + one cross-cutting = 3
    assert len(docs) == 3
    by_industry = {d.industry: d for d in docs}
    assert "制造" in by_industry
    assert "批零" in by_industry
    # cross_cutting → industry=None
    assert None in by_industry

    manu = by_industry["制造"]
    assert manu.title == "制造业测试样本"
    # 应该有 3 个 chunk（应收账款 / 存货周转 / 关联方风险）
    assert len(manu.chunks) >= 3
    chunk_industries = {c.industry for c in manu.chunks}
    assert chunk_industries == {"制造"}


def test_load_corpus_skips_underscore_files(tmp_path: Path):
    (tmp_path / "industries").mkdir()
    (tmp_path / "industries" / "制造.md").write_text("# 制造\n\n内容\n", encoding="utf-8")
    (tmp_path / "industries" / "_skip.md").write_text("# 应跳过\n", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert len(docs) == 1
    assert docs[0].industry == "制造"


def test_load_corpus_missing_root_returns_empty(tmp_path: Path):
    docs = load_corpus(tmp_path / "doesnotexist")
    assert docs == []


def test_load_corpus_only_picks_up_industries_subdir_naming(tmp_path: Path):
    # File at top level (not under industries/) → industry=None
    (tmp_path / "top.md").write_text("# top doc\n\ntop body", encoding="utf-8")
    # File under industries/ → industry=<filename>
    (tmp_path / "industries").mkdir()
    (tmp_path / "industries" / "建筑.md").write_text("# 建筑\n\n建筑业内容", encoding="utf-8")
    # File under arbitrary other subdir → industry=None
    (tmp_path / "cross_cutting").mkdir()
    (tmp_path / "cross_cutting" / "通用.md").write_text("# 通用\n\n通用内容", encoding="utf-8")

    docs = load_corpus(tmp_path)
    by_id = {d.doc_id: d for d in docs}
    assert by_id["top"].industry is None
    assert by_id["industries/建筑"].industry == "建筑"
    assert by_id["cross_cutting/通用"].industry is None
