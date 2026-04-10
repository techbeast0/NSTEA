"""Text embedder for RAG — uses sentence-transformers for local embedding."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a sentence-transformer model for embedding clinical text."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None  # Lazy load

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: List of text strings.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._ensure_model()
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._model.get_sentence_embedding_dimension()
