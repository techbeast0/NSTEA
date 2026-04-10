"""Guideline search tool: search clinical guidelines via RAG vector retrieval.

Wraps the VectorStore search as a callable function for agent pipelines.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from nstea.config import settings
from nstea.retrieval.embedder import Embedder
from nstea.retrieval.vector_store import VectorStore

INDEX_PATH = settings.project_root / "data" / "vector_index.pkl"

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    """Lazily load the pre-built vector index or build one on-the-fly."""
    global _store
    if _store is not None:
        return _store

    embedder = Embedder()
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _store = VectorStore(embedder)
        _store._chunks = data["chunks"]
        _store._vectors = data["vectors"]
    else:
        # Build on-the-fly from guidelines dir
        from nstea.retrieval import load_directory
        guidelines_dir = settings.project_root / "data" / "guidelines"
        chunks = load_directory(guidelines_dir)
        _store = VectorStore(embedder)
        _store.index(chunks)
    return _store


def search_guidelines(query: str) -> str:
    """Search clinical guidelines for information relevant to the query.

    Use this tool to find evidence-based guidance for treatment decisions,
    drug dosing, contraindications, and clinical management protocols.

    Args:
        query: Natural language clinical question, e.g.
               "treatment for UTI in patient with sulfa allergy"
               "beta-blocker in asthma COPD management"
               "warfarin drug interactions with antibiotics"

    Returns:
        Relevant guideline excerpts with source attribution and relevance scores.
    """
    store = _get_store()
    results = store.search(query, top_k=5, score_threshold=0.25)

    if not results:
        return (
            "No matching guidelines found. Base your recommendation on general "
            "clinical knowledge and explicitly note the absence of guideline support."
        )

    lines = [f"Found {len(results)} relevant guideline excerpt(s):\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"--- Guideline {i} (source: {r.chunk.source}, relevance: {r.score:.2f}) ---")
        lines.append(r.chunk.text)
        lines.append("")

    return "\n".join(lines)
