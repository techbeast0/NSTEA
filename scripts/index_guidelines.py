"""Index clinical guideline documents into the vector store for RAG retrieval.

Usage:
    python scripts/index_guidelines.py [--guidelines-dir data/guidelines]
"""

import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstea.retrieval import load_directory  # noqa: E402
from nstea.retrieval.embedder import Embedder  # noqa: E402
from nstea.retrieval.vector_store import VectorStore  # noqa: E402


INDEX_PATH = PROJECT_ROOT / "data" / "vector_index.pkl"


def build_index(guidelines_dir: Path | None = None) -> VectorStore:
    """Load guidelines, embed, and build a VectorStore index."""
    if guidelines_dir is None:
        guidelines_dir = PROJECT_ROOT / "data" / "guidelines"

    print(f"Loading guideline documents from {guidelines_dir} ...")
    chunks = load_directory(guidelines_dir)
    print(f"  → {len(chunks)} chunks from {len(set(c.source for c in chunks))} files")

    print("Initializing embedder (sentence-transformers) ...")
    embedder = Embedder()

    print("Building vector store ...")
    store = VectorStore(embedder)
    n = store.index(chunks)
    print(f"  → Indexed {n} chunks (dim={embedder.dimension})")

    return store


def save_index(store: VectorStore, path: Path = INDEX_PATH) -> None:
    """Persist the vector store to disk."""
    with open(path, "wb") as f:
        pickle.dump({"chunks": store._chunks, "vectors": store._vectors}, f)
    print(f"Index saved to {path}")


def load_index(embedder: Embedder, path: Path = INDEX_PATH) -> VectorStore:
    """Load a persisted vector store from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    store = VectorStore(embedder)
    store._chunks = data["chunks"]
    store._vectors = data["vectors"]
    return store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index clinical guidelines for RAG")
    parser.add_argument("--guidelines-dir", type=Path, default=None)
    args = parser.parse_args()

    store = build_index(args.guidelines_dir)
    save_index(store)

    # Quick test
    print("\n--- Quick search test ---")
    results = store.search("aspirin allergy management acute coronary syndrome", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.source}: {r.chunk.text[:80]}...")
