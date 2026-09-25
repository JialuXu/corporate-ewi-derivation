"""Corpus data structures.

A `CorpusDocument` is one markdown file. We split it into `CorpusChunk`s by
heading boundary — each chunk is one section that the retriever ranks
independently. Chunks inherit `industry` from the parent document's
classification (set by the directory layout in `corpus/`).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CorpusChunk(BaseModel):
    """One retrievable text section."""

    chunk_id: str   # f"{doc_id}#{joined_heading_path}"
    text: str       # markdown-stripped plain text
    industry: str | None = None
    headings: list[str] = Field(default_factory=list)


class CorpusDocument(BaseModel):
    """One markdown file under corpus/."""

    doc_id: str     # path relative to corpus root, no extension
    industry: str | None = None
    title: str
    chunks: list[CorpusChunk] = Field(default_factory=list)
