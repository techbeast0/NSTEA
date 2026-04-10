"""Batch computation scheduler for temporal embeddings.

Computes embeddings for multiple patients in batch (e.g., daily cron job).
In production, this would be a Celery task or scheduled job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import structlog

from nstea.models.patient import PatientInput
from nstea.temporal.embedding_cache import EmbeddingCache
from nstea.temporal.graph_builder import PatientGraphBuilder
from nstea.temporal.temporal_encoder import TemporalEncoder

logger = structlog.stdlib.get_logger(__name__)


@dataclass
class BatchResult:
    """Result of a batch embedding update."""

    total_patients: int = 0
    updated: int = 0
    cache_hits: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0
    error_details: list[str] = field(default_factory=list)


class BatchEmbeddingUpdater:
    """Batch computation of patient temporal embeddings.

    Usage:
        updater = BatchEmbeddingUpdater()
        result = updater.update_batch(patients)
    """

    def __init__(
        self,
        cache: EmbeddingCache | None = None,
        graph_builder: PatientGraphBuilder | None = None,
        encoder: TemporalEncoder | None = None,
        embedding_dim: int = 64,
    ):
        self.cache = cache or EmbeddingCache()
        self.graph_builder = graph_builder or PatientGraphBuilder()
        self.encoder = encoder or TemporalEncoder()
        self.embedding_dim = embedding_dim

    def compute_single(
        self,
        patient: PatientInput,
        reference_date: date | None = None,
        force_recompute: bool = False,
    ) -> tuple[list[float], list[str], dict]:
        """Compute embedding for a single patient.

        Returns:
            Tuple of (embedding, insights, graph_summary)
        """
        # Check cache first
        cache_key = self.cache.compute_key(
            patient.patient_id,
            {"history": [e.model_dump() for e in patient.history]},
        )

        if not force_recompute:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached.embedding, cached.insights, cached.graph_summary

        # Build graph
        graph = self.graph_builder.build_graph(patient)

        # Compute embedding
        embedding = self.encoder.compute_embedding(
            graph,
            reference_date=reference_date,
            embedding_dim=self.embedding_dim,
        )

        # Generate insights
        insights = self.encoder.get_top_insights(graph, reference_date)

        # Graph summary for cache/API
        graph_summary = {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "date_range": [
                d.isoformat() if d else None
                for d in graph.date_range
            ],
        }

        # Cache result
        self.cache.put(cache_key, embedding, insights, graph_summary)

        return embedding, insights, graph_summary

    def update_batch(
        self,
        patients: list[PatientInput],
        reference_date: date | None = None,
        force_recompute: bool = False,
    ) -> BatchResult:
        """Compute embeddings for a batch of patients.

        Args:
            patients: List of patients to process.
            reference_date: Reference date for temporal encoding.
            force_recompute: If True, ignore cache and recompute all.

        Returns:
            BatchResult with statistics.
        """
        start = time.time()
        result = BatchResult(total_patients=len(patients))

        for patient in patients:
            try:
                cache_key = self.cache.compute_key(
                    patient.patient_id,
                    {"history": [e.model_dump() for e in patient.history]},
                )

                if not force_recompute:
                    cached = self.cache.get(cache_key)
                    if cached is not None:
                        result.cache_hits += 1
                        continue

                self.compute_single(patient, reference_date, force_recompute=True)
                result.updated += 1

            except Exception as e:
                result.errors += 1
                result.error_details.append(f"{patient.patient_id}: {e}")
                logger.warning(
                    "batch_embedding_error",
                    patient_id=patient.patient_id,
                    error=str(e),
                )

        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "batch_update_complete",
            total=result.total_patients,
            updated=result.updated,
            cache_hits=result.cache_hits,
            errors=result.errors,
            elapsed=result.elapsed_seconds,
        )

        return result
