"""Unit tests for the RAG retrieval pipeline."""

import pytest
from nstea.retrieval import load_and_chunk, load_directory, DocumentChunk
from nstea.retrieval.embedder import Embedder
from nstea.retrieval.vector_store import VectorStore
from nstea.config import settings


GUIDELINES_DIR = settings.project_root / "data" / "guidelines"


class TestDocumentLoader:
    def test_load_single_file(self):
        acs_file = GUIDELINES_DIR / "acs_management.txt"
        if not acs_file.exists():
            pytest.skip("ACS guideline file not found")
        chunks = load_and_chunk(acs_file)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.source == "acs_management" for c in chunks)

    def test_load_directory(self):
        if not GUIDELINES_DIR.exists():
            pytest.skip("Guidelines directory not found")
        chunks = load_directory(GUIDELINES_DIR)
        assert len(chunks) > 0
        sources = set(c.source for c in chunks)
        assert len(sources) >= 2  # we have multiple guideline files

    def test_chunk_ids_unique(self):
        if not GUIDELINES_DIR.exists():
            pytest.skip("Guidelines directory not found")
        chunks = load_directory(GUIDELINES_DIR)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        return Embedder()

    def test_embed_single(self, embedder):
        vec = embedder.embed_single("aspirin allergy management")
        assert vec.shape == (embedder.dimension,)

    def test_embed_batch(self, embedder):
        texts = ["hypertension treatment", "diabetes management", "UTI antibiotics"]
        vecs = embedder.embed(texts)
        assert vecs.shape == (3, embedder.dimension)

    def test_normalized_vectors(self, embedder):
        import numpy as np
        vec = embedder.embed_single("test query")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01, f"Expected unit vector, got norm={norm}"


class TestVectorStore:
    @pytest.fixture(scope="class")
    def store(self):
        embedder = Embedder()
        chunks = load_directory(GUIDELINES_DIR)
        store = VectorStore(embedder)
        store.index(chunks)
        return store

    def test_index_count(self, store):
        assert store.count > 0

    def test_search_returns_results(self, store):
        results = store.search("warfarin drug interactions bleeding risk")
        assert len(results) > 0
        assert results[0].score > 0.3

    def test_search_relevance_order(self, store):
        results = store.search("pregnancy contraindicated medications")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_search_acs_query(self, store):
        results = store.search("acute coronary syndrome STEMI treatment")
        assert len(results) > 0
        # At least one result should be from ACS guidelines
        sources = [r.chunk.source for r in results]
        assert any("acs" in s.lower() for s in sources)

    def test_search_empty_store(self):
        embedder = Embedder()
        empty_store = VectorStore(embedder)
        results = empty_store.search("anything")
        assert results == []
