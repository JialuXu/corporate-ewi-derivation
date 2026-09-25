"""Corpus loader — walk markdown files under corpus/ and chunk by heading.

Directory layout convention (must match the curated corpus):

    corpus/
    ├── _meta.yaml                 (optional; schema documentation)
    ├── industries/
    │   ├── 制造.md       → industry="制造"
    │   ├── 批零.md       → industry="批零"
    │   └── 建筑.md       → industry="建筑"
    ├── cross_cutting/             → industry=None (applies to all industries)
    │   └── ...
    └── pitfalls/                  → industry=None
        └── ...

Industries are derived from the FILENAME under `industries/` (without
`.md`). Files under any other top-level directory get `industry=None`.

Empty directories are tolerated — the corpus is bootstrapped incrementally,
and an empty `cross_cutting/` on day 1 is the expected state.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .corpus import CorpusChunk, CorpusDocument

logger = logging.getLogger(__name__)

INDUSTRIES_DIR = "industries"

_HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"\*([^*]+)\*")
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_LIST_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Remove markdown syntax leaving plain text for TF-IDF.

    We intentionally KEEP heading text (just drop the `#` prefix) so terms
    in section titles still contribute to retrieval.
    """
    text = _FRONTMATTER_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _LIST_BULLET_RE.sub("", text)
    text = _ORDERED_LIST_RE.sub("", text)
    # Drop leading '#' from headings — keep the text.
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_markdown(
    text: str,
    *,
    max_chars: int = 800,
) -> list[tuple[list[str], str]]:
    """Split markdown by heading boundaries; return [(heading_path, body)].

    `heading_path` is the list of section titles from `#` down to the
    deepest preceding heading at that point in the document. `max_chars` is a
    soft limit: longer sections are kept as a single chunk to avoid splitting
    mid-paragraph and breaking retrieval.
    """
    text = _FRONTMATTER_RE.sub("", text)
    sections: list[tuple[list[str], str]] = []

    # Walk headings + the bodies between them.
    headings = list(_HEADING_RE.finditer(text))
    if not headings:
        body = _strip_markdown(text)
        if body:
            sections.append(([], body))
        return sections

    stack: list[tuple[int, str]] = []  # (level, title)
    # Capture pre-heading preamble if any.
    if headings[0].start() > 0:
        pre = text[: headings[0].start()].strip()
        body = _strip_markdown(pre)
        if body:
            sections.append(([], body))
    for i, m in enumerate(headings):
        level = len(m.group(1))
        title = m.group(2).strip()
        # Pop any deeper-or-equal levels off the stack.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [t for _, t in stack]
        # Body is everything until the next heading.
        body_start = m.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        raw_body = text[body_start:body_end]
        body = _strip_markdown(raw_body)
        if body:
            # Include the heading text itself in the chunk body for retrieval.
            heading_text = " ".join(heading_path)
            full = f"{heading_text} {body}"
            sections.append((heading_path[:], full))
    return sections


def _industry_for_doc(rel_path: Path) -> str | None:
    """Return industry tag based on directory placement."""
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0] == INDUSTRIES_DIR:
        return rel_path.stem
    return None


def _doc_title_from_md(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def load_corpus(root: Path) -> list[CorpusDocument]:
    """Load all `.md` files under `root` (recursively) into CorpusDocuments.

    Missing root, empty subdirectories, or zero-document corpora all return
    `[]`. Callers should treat that as cold-start.
    """
    if not root.exists():
        logger.info("corpus root does not exist: %s — returning empty list", root)
        return []
    docs: list[CorpusDocument] = []
    for md_path in sorted(root.rglob("*.md")):
        if md_path.name.startswith("_"):
            continue
        rel = md_path.relative_to(root)
        industry = _industry_for_doc(rel)
        try:
            text = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logger.warning("skipping non-utf8 corpus file %s: %s", md_path, e)
            continue
        title = _doc_title_from_md(text, fallback=md_path.stem)
        doc_id = str(rel.with_suffix(""))
        chunks: list[CorpusChunk] = []
        for headings, body in chunk_markdown(text):
            chunk_id = f"{doc_id}#{'/'.join(headings) or 'root'}"
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    text=body,
                    industry=industry,
                    headings=headings,
                )
            )
        docs.append(CorpusDocument(doc_id=doc_id, industry=industry, title=title, chunks=chunks))
    return docs
