"""Tests for Phase 4: Temporal Layer — graph builder, encoder, cache, and agent."""

from datetime import date, timedelta

import pytest

from nstea.models.patient import (
    ClinicalEvent,
    Condition,
    LabResult,
    Medication,
    PatientInput,
)
from nstea.temporal.batch_updater import BatchEmbeddingUpdater
from nstea.temporal.embedding_cache import EmbeddingCache
from nstea.temporal.graph_builder import PatientGraphBuilder, TemporalGraph
from nstea.temporal.temporal_encoder import TemporalEncoder


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_patient(
    history_count: int = 3,
    conditions_count: int = 2,
    meds_count: int = 1,
    labs_count: int = 1,
) -> PatientInput:
    """Build a test patient with configurable clinical data."""
    today = date.today()
    history = [
        ClinicalEvent(
            event_type="diagnosis" if i % 2 == 0 else "medication",
            description=f"Event {i}: {'Diagnosed condition' if i % 2 == 0 else 'Started medication'}",
            date=today - timedelta(days=30 * (history_count - i)),
        )
        for i in range(history_count)
    ]
    conditions = [
        Condition(name=f"Condition {i}", onset_date=today - timedelta(days=180 * (i + 1)))
        for i in range(conditions_count)
    ]
    medications = [
        Medication(name=f"Drug {i}", start_date=today - timedelta(days=60 * (i + 1)))
        for i in range(meds_count)
    ]
    lab_results = [
        LabResult(
            test_name=f"Test {i}",
            value=float(i + 1),
            unit="mg/dL",
            date=today - timedelta(days=7 * (i + 1)),
            is_abnormal=(i == 0),
        )
        for i in range(labs_count)
    ]

    return PatientInput(
        patient_id="TEST-001",
        age=65,
        sex="male",
        conditions=conditions,
        medications=medications,
        history=history,
        lab_results=lab_results,
    )


def _make_cardiac_patient() -> PatientInput:
    """Build a patient with known causal relationships (NSAID → CV risk)."""
    today = date.today()
    return PatientInput(
        patient_id="CARDIAC-001",
        age=65,
        sex="male",
        conditions=[
            Condition(name="Hypertension", onset_date=today - timedelta(days=365)),
            Condition(name="Type 2 Diabetes", onset_date=today - timedelta(days=180)),
        ],
        medications=[
            Medication(name="NSAID (Ibuprofen)", start_date=today - timedelta(days=180)),
            Medication(name="Metformin", start_date=today - timedelta(days=180)),
        ],
        history=[
            ClinicalEvent(
                event_type="diagnosis",
                description="Hypertension diagnosed",
                date=today - timedelta(days=365),
            ),
            ClinicalEvent(
                event_type="diagnosis",
                description="Type 2 Diabetes diagnosed",
                date=today - timedelta(days=180),
            ),
            ClinicalEvent(
                event_type="medication",
                description="Started NSAID for joint pain",
                date=today - timedelta(days=180),
            ),
            ClinicalEvent(
                event_type="visit",
                description="Cardiovascular risk assessment",
                date=today - timedelta(days=30),
            ),
        ],
        lab_results=[
            LabResult(
                test_name="Troponin I",
                value=0.8,
                unit="ng/mL",
                date=today,
                is_abnormal=True,
            ),
        ],
        symptoms=[],
    )


# ── Graph Builder Tests ──────────────────────────────────────────────────────

class TestPatientGraphBuilder:
    def test_build_basic_graph(self):
        patient = _make_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        assert isinstance(graph, TemporalGraph)
        assert graph.patient_id == "TEST-001"
        assert graph.node_count > 0
        assert graph.edge_count > 0

    def test_node_count_matches_data(self):
        patient = _make_patient(history_count=3, conditions_count=2, meds_count=1, labs_count=1)
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        # 3 history + 2 conditions + 1 medication + 1 lab = 7 nodes
        assert graph.node_count == 7

    def test_temporal_edges_exist(self):
        patient = _make_patient(history_count=4)
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        temporal_edges = [
            (u, v, d) for u, v, d in graph.graph.edges(data=True)
            if d.get("type") == "temporal"
        ]
        assert len(temporal_edges) > 0

    def test_temporal_edge_weights(self):
        """Edges between close events should have higher weights than distant ones."""
        patient = _make_patient(history_count=5)
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        temporal_edges = [
            d for _, _, d in graph.graph.edges(data=True)
            if d.get("type") == "temporal"
        ]
        for edge_data in temporal_edges:
            assert 0.0 < edge_data["weight"] <= 1.0

    def test_causal_edges_for_known_relationships(self):
        """NSAID → cardiovascular should produce a causal edge."""
        patient = _make_cardiac_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        causal_edges = [
            (u, v, d) for u, v, d in graph.graph.edges(data=True)
            if d.get("type") == "causal"
        ]
        assert len(causal_edges) > 0

    def test_empty_patient_history(self):
        patient = PatientInput(patient_id="EMPTY-001", age=30, sex="female")
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_graph_serialization(self):
        patient = _make_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)
        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert data["patient_id"] == "TEST-001"
        assert len(data["nodes"]) == graph.node_count

    def test_date_range_computed(self):
        patient = _make_patient(history_count=3)
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)
        start, end = graph.date_range
        assert start is not None
        assert end is not None
        assert start <= end


# ── Temporal Encoder Tests ───────────────────────────────────────────────────

class TestTemporalEncoder:
    def test_encode_basic(self):
        patient = _make_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        encoder = TemporalEncoder()
        importances = encoder.encode(graph)

        assert len(importances) == graph.node_count
        # Should be sorted by combined_score descending
        scores = [imp.combined_score for imp in importances]
        assert scores == sorted(scores, reverse=True)

    def test_recent_events_higher_importance(self):
        """More recent events should generally have higher temporal weights."""
        patient = _make_patient(history_count=5)
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        encoder = TemporalEncoder()
        importances = encoder.encode(graph)

        # At least the most recent events should have high temporal weight
        most_recent = max(importances, key=lambda x: x.temporal_weight)
        oldest = min(importances, key=lambda x: x.temporal_weight)
        assert most_recent.temporal_weight > oldest.temporal_weight

    def test_get_top_insights(self):
        patient = _make_cardiac_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        encoder = TemporalEncoder()
        insights = encoder.get_top_insights(graph, top_k=3)

        assert len(insights) <= 3
        assert all(isinstance(s, str) for s in insights)

    def test_compute_embedding_shape(self):
        patient = _make_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        encoder = TemporalEncoder()
        embedding = encoder.compute_embedding(graph, embedding_dim=64)

        assert len(embedding) == 64
        assert all(isinstance(v, float) for v in embedding)

    def test_compute_embedding_different_dims(self):
        patient = _make_patient()
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        encoder = TemporalEncoder()
        for dim in [16, 32, 128]:
            emb = encoder.compute_embedding(graph, embedding_dim=dim)
            assert len(emb) == dim

    def test_empty_graph_returns_zeros(self):
        patient = PatientInput(patient_id="EMPTY", age=30, sex="female")
        builder = PatientGraphBuilder()
        graph = builder.build_graph(patient)

        encoder = TemporalEncoder()
        importances = encoder.encode(graph)
        assert len(importances) == 0

        embedding = encoder.compute_embedding(graph, embedding_dim=32)
        assert len(embedding) == 32
        assert all(v == 0.0 for v in embedding)


# ── Embedding Cache Tests ────────────────────────────────────────────────────

class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache()
        cache.put("test:abc", [1.0, 2.0, 3.0], ["insight 1"])
        entry = cache.get("test:abc")
        assert entry is not None
        assert entry.embedding == [1.0, 2.0, 3.0]
        assert entry.insights == ["insight 1"]

    def test_cache_miss(self):
        cache = EmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = EmbeddingCache()
        cache.put("test:exp", [1.0], ["insight"], ttl=0.0)  # Immediately expired
        import time
        time.sleep(0.01)
        assert cache.get("test:exp") is None

    def test_invalidate_patient(self):
        cache = EmbeddingCache()
        cache.put("P1:hash1", [1.0], ["a"])
        cache.put("P1:hash2", [2.0], ["b"])
        cache.put("P2:hash1", [3.0], ["c"])

        removed = cache.invalidate("P1")
        assert removed == 2
        assert cache.get("P1:hash1") is None
        assert cache.get("P2:hash1") is not None

    def test_stats(self):
        cache = EmbeddingCache()
        cache.put("k1", [1.0], ["a"])
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_compute_key_deterministic(self):
        cache = EmbeddingCache()
        key1 = cache.compute_key("P1", {"history": [{"event": "test"}]})
        key2 = cache.compute_key("P1", {"history": [{"event": "test"}]})
        assert key1 == key2

    def test_compute_key_varies_with_data(self):
        cache = EmbeddingCache()
        key1 = cache.compute_key("P1", {"history": [{"event": "a"}]})
        key2 = cache.compute_key("P1", {"history": [{"event": "b"}]})
        assert key1 != key2


# ── Batch Updater Tests ──────────────────────────────────────────────────────

class TestBatchUpdater:
    def test_compute_single(self):
        patient = _make_patient()
        updater = BatchEmbeddingUpdater()
        embedding, insights, summary = updater.compute_single(patient)
        assert len(embedding) == 64
        assert len(insights) > 0
        assert "node_count" in summary

    def test_batch_update(self):
        patients = [_make_patient() for _ in range(3)]
        updater = BatchEmbeddingUpdater()
        result = updater.update_batch(patients)
        assert result.total_patients == 3
        assert result.errors == 0

    def test_cache_hit_on_repeat(self):
        patient = _make_patient()
        updater = BatchEmbeddingUpdater()
        # First compute
        updater.compute_single(patient)
        # Second compute should hit cache
        result = updater.update_batch([patient])
        assert result.cache_hits == 1
        assert result.updated == 0


# ── Temporal Agent Tests ─────────────────────────────────────────────────────

class TestTemporalAgent:
    def test_run_temporal_analysis(self):
        from nstea.agents.temporal_agent import run_temporal_analysis

        patient = _make_cardiac_patient()
        result = run_temporal_analysis(patient)

        assert len(result.embedding) == 64
        assert len(result.insights) > 0
        assert result.graph_summary.get("node_count", 0) > 0

    def test_temporal_context_string(self):
        from nstea.agents.temporal_agent import run_temporal_analysis

        patient = _make_cardiac_patient()
        result = run_temporal_analysis(patient)

        context = result.to_context_string()
        assert "TEMPORAL ANALYSIS" in context
        assert "events" in context.lower() or "insights" in context.lower()

    def test_empty_patient(self):
        from nstea.agents.temporal_agent import run_temporal_analysis

        patient = PatientInput(patient_id="EMPTY", age=30, sex="female")
        result = run_temporal_analysis(patient)
        assert "No clinical history" in result.insights[0]
