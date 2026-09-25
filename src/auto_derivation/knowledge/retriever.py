"""TF-IDF retriever over CorpusChunks.

Chinese text uses character n-gram analyzers (`char_wb`, ngram_range=(2,4)),
so business terms like "应收账款 / 地方融资平台" are recognised as
overlapping char windows without word segmentation.

Industry filtering happens BEFORE TF-IDF: we restrict the candidate pool to
chunks whose `industry` is None (cross-cutting) or matches the query's
industry, then rank by similarity. This is faster and avoids spurious
cross-industry hits.

A retriever fit on zero chunks is allowed — `query()` then returns []. The
suspicion scorer relies on this for cold-start.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import CorpusChunk

logger = logging.getLogger(__name__)


class TfidfRetriever:
    """Local TF-IDF retriever. Stateless apart from the fitted vectorizer."""

    def __init__(
        self,
        chunks: list[CorpusChunk],
        *,
        analyzer: Literal["char_wb", "word", "char"] = "char_wb",
        ngram_range: tuple[int, int] = (2, 4),
    ):
        self.chunks: list[CorpusChunk] = list(chunks)
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None  # scipy sparse matrix; type kept open for sklearn versions

    def fit(self) -> None:
        """Fit the TF-IDF vectorizer on all chunk texts. Idempotent."""
        if not self.chunks:
            logger.info("TfidfRetriever.fit: 0 chunks — leaving vectorizer unfit")
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=False,   # Chinese doesn't need it; preserves any ASCII tokens
        )
        texts = [c.text for c in self.chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def query(
        self,
        q: str,
        *,
        industry: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[CorpusChunk, float]]:
        """Return top-k (chunk, similarity) pairs. Filters by industry first.

        Empty `q`, empty corpus, or unfit vectorizer all return []."""
        if not q.strip() or self._vectorizer is None or self._matrix is None:
            return []
        candidate_idx = [
            i for i, c in enumerate(self.chunks)
            if c.industry is None or (industry is None or c.industry == industry)
        ]
        if not candidate_idx:
            return []
        q_vec = self._vectorizer.transform([q])
        # scipy sparse matrices accept ndarray indices at runtime but the
        # type stubs only declare scalar/slice — np.asarray sidesteps this.
        sub_matrix = self._matrix[np.asarray(candidate_idx)]
        sims = cosine_similarity(q_vec, sub_matrix).ravel()
        # Argsort descending, take top_k.
        order = sims.argsort()[::-1][:top_k]
        out: list[tuple[CorpusChunk, float]] = []
        for j in order:
            score = float(sims[j])
            if score <= 0.0:
                continue
            out.append((self.chunks[candidate_idx[j]], score))
        return out

    def __len__(self) -> int:
        return len(self.chunks)
