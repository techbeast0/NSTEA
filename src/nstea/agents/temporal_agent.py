"""Temporal Agent — fetches or computes T-GNN embeddings for patients.

Integrates into the orchestrator pipeline, running in parallel with RAG retrieval.
Uses the embedding cache for <50ms cached lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import structlog

from nstea.models.patient import PatientInput
from nstea.temporal.batch_updater import BatchEmbeddingUpdater
from nstea.temporal.embedding_cache import EmbeddingCache
from nstea.temporal.graph_builder import PatientGraphBuilder
from nstea.temporal.temporal_encoder import TemporalEncoder

logger = structlog.stdlib.get_logger(__name__)

# Module-level singleton (lazy init)
_temporal_engine: BatchEmbeddingUpdater | None = None
_embedding_cache: EmbeddingCache | None = None


def _get_engine() -> BatchEmbeddingUpdater:
    """Get or create the temporal engine singleton."""
    global _temporal_engine, _embedding_cache
    if _temporal_engine is None:
        _embedding_cache = EmbeddingCache(default_ttl=86400)
        _temporal_engine = BatchEmbeddingUpdater(
            cache=_embedding_cache,
            graph_builder=PatientGraphBuilder(decay_lambda=0.01),
            encoder=TemporalEncoder(decay_lambda=0.005),
            embedding_dim=64,
        )
    return _temporal_engine


@dataclass
class TemporalResult:
    """Result from temporal analysis."""

    embedding: list[float] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    graph_summary: dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False

    def to_context_string(self) -> str:
        """Format temporal insights as context for LLM injection."""
        if not self.insights:
            return "TEMPORAL ANALYSIS: No clinical history available for temporal analysis."

        lines = ["=== TEMPORAL ANALYSIS ==="]
        lines.append(f"Patient graph: {self.graph_summary.get('node_count', 0)} events, "
                      f"{self.graph_summary.get('edge_count', 0)} relationships")

        date_range = self.graph_summary.get("date_range", [None, None])
        if date_range[0] and date_range[1]:
            lines.append(f"History span: {date_range[0]} to {date_range[1]}")

        lines.append("")
        lines.append("Key temporal insights (most important first):")
        for i, insight in enumerate(self.insights, 1):
            lines.append(f"  {i}. {insight}")

        return "\n".join(lines)


def run_temporal_analysis(
    patient: PatientInput,
    reference_date: date | None = None,
) -> TemporalResult:
    """Run temporal analysis for a patient.

    This is the main entry point called by the orchestrator.
    Computes or retrieves cached temporal embedding and insights.
    """
    engine = _get_engine()

    # Check if patient has history data worth analyzing
    has_history = bool(patient.history or patient.conditions or patient.lab_results)
    if not has_history:
        logger.info("temporal_skip", patient_id=patient.patient_id, reason="no_history")
        return TemporalResult(
            insights=["No clinical history available for temporal analysis."],
        )

    try:
        # Check cache
        cache_key = engine.cache.compute_key(
            patient.patient_id,
            {"history": [e.model_dump() for e in patient.history]},
        )
        cached = engine.cache.get(cache_key)

        if cached is not None:
            logger.info("temporal_cache_hit", patient_id=patient.patient_id)
            return TemporalResult(
                embedding=cached.embedding,
                insights=cached.insights,
                graph_summary=cached.graph_summary,
                from_cache=True,
            )

        # Compute fresh
        embedding, insights, graph_summary = engine.compute_single(
            patient, reference_date=reference_date
        )

        logger.info(
            "temporal_computed",
            patient_id=patient.patient_id,
            node_count=graph_summary.get("node_count", 0),
            insights_count=len(insights),
        )

        return TemporalResult(
            embedding=embedding,
            insights=insights,
            graph_summary=graph_summary,
            from_cache=False,
        )

    except Exception as e:
        logger.exception("temporal_analysis_failed", patient_id=patient.patient_id)
        return TemporalResult(
            insights=[f"Temporal analysis failed: {type(e).__name__}"],
        )


def get_cache_stats() -> dict[str, Any]:
    """Return temporal embedding cache statistics."""
    engine = _get_engine()
    return engine.cache.stats
