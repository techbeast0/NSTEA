"""In-memory vector store with optional Qdrant backend for RAG retrieval.

Phase 1 starts with a simple numpy-based store that can be swapped for Qdrant
when Docker infra is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nstea.retrieval import DocumentChunk
from nstea.retrieval.embedder import Embedder


@dataclass
class SearchResult:
    """A single retrieval result."""

    chunk: DocumentChunk
    score: float


class VectorStore:
    """Numpy-based in-memory vector store (upgradeable to Qdrant)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._chunks: list[DocumentChunk] = []
        self._vectors: np.ndarray | None = None

    def index(self, chunks: list[DocumentChunk]) -> int:
        """Index a list of document chunks.

        Args:
            chunks: List of DocumentChunk to embed and store.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)

        if self._vectors is None:
            self._chunks = chunks
            self._vectors = vectors
        else:
            self._chunks.extend(chunks)
            self._vectors = np.vstack([self._vectors, vectors])

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Search for the most relevant chunks given a query.

        Args:
            query: Natural language query.
            top_k: Maximum number of results.
            score_threshold: Minimum cosine similarity to include.

        Returns:
            List of SearchResult sorted by descending score.
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        q_vec = self.embedder.embed_single(query)
        # Cosine similarity (vectors are already normalized)
        scores = self._vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= score_threshold:
                results.append(SearchResult(chunk=self._chunks[idx], score=score))
        return results

    @property
    def count(self) -> int:
        return len(self._chunks)
