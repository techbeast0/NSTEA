"""In-memory embedding cache for T-GNN patient embeddings.

Provides a dict-backed cache with TTL expiry. In production, swap for Redis.
Cache key: "{patient_id}:{data_hash}" where data_hash is computed from patient history.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """Single cached embedding with metadata."""

    embedding: list[float]
    insights: list[str]
    graph_summary: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 86400  # 24 hours default

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class EmbeddingCache:
    """In-memory cache for patient temporal embeddings.

    Thread-safe for single-process usage. For multi-process, use Redis backend.
    """

    def __init__(self, default_ttl: float = 86400):
        self._cache: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def compute_key(self, patient_id: str, patient_data: dict[str, Any]) -> str:
        """Compute cache key from patient ID + history hash."""
        # Hash the history portion for change detection
        history_str = json.dumps(
            patient_data.get("history", []),
            sort_keys=True,
            default=str,
        )
        data_hash = hashlib.sha256(history_str.encode()).hexdigest()[:12]
        return f"{patient_id}:{data_hash}"

    def get(self, key: str) -> CacheEntry | None:
        """Retrieve cached embedding. Returns None if miss or expired."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry

    def put(
        self,
        key: str,
        embedding: list[float],
        insights: list[str],
        graph_summary: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Store embedding in cache."""
        self._cache[key] = CacheEntry(
            embedding=embedding,
            insights=insights,
            graph_summary=graph_summary or {},
            ttl_seconds=self.default_ttl if ttl is None else ttl,
        )

    def invalidate(self, patient_id: str) -> int:
        """Invalidate all cache entries for a patient. Returns count removed."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"{patient_id}:")]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
        return len(expired)
