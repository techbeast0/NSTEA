"""Temporal encoder — computes time-decay importance weights for graph nodes.

Uses exponential decay to weight recent events more heavily,
with type-based multipliers for clinical importance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import networkx as nx

from nstea.temporal.graph_builder import TemporalGraph

# Clinical event type importance multipliers
TYPE_MULTIPLIERS: dict[str, float] = {
    "diagnosis": 1.2,
    "medication": 1.0,
    "lab": 0.9,
    "procedure": 1.1,
    "visit": 0.5,
    "imaging": 0.8,
}


@dataclass
class NodeImportance:
    """Importance score for a single graph node."""

    node_id: str
    description: str
    temporal_weight: float  # Time-decay component
    type_weight: float  # Event type component
    centrality_weight: float  # Graph centrality component
    combined_score: float  # Final weighted score


class TemporalEncoder:
    """Computes importance weights for graph nodes relative to a reference date.

    Combine three signals:
    1. Temporal decay: recent events matter more
    2. Type importance: diagnoses > labs > visits
    3. Graph centrality: well-connected nodes are more relevant
    """

    def __init__(
        self,
        decay_lambda: float = 0.005,
        temporal_weight: float = 0.5,
        type_weight: float = 0.3,
        centrality_weight: float = 0.2,
    ):
        self.decay_lambda = decay_lambda
        self.temporal_w = temporal_weight
        self.type_w = type_weight
        self.centrality_w = centrality_weight

    def encode(
        self,
        graph: TemporalGraph,
        reference_date: date | None = None,
    ) -> list[NodeImportance]:
        """Compute importance weights for all nodes in the graph.

        Args:
            graph: Patient temporal graph.
            reference_date: Date to compute recency from (defaults to today).

        Returns:
            List of NodeImportance, sorted by combined_score descending.
        """
        if reference_date is None:
            reference_date = date.today()

        G = graph.graph
        if G.number_of_nodes() == 0:
            return []

        # Compute graph centrality (degree-based for directed graph)
        try:
            centrality = nx.degree_centrality(G)
        except Exception:
            centrality = {n: 0.0 for n in G.nodes()}

        results: list[NodeImportance] = []
        for node_id, data in G.nodes(data=True):
            # 1. Temporal decay
            node_date = data.get("date", reference_date)
            days_ago = max((reference_date - node_date).days, 0)
            temporal = math.exp(-self.decay_lambda * days_ago)

            # 2. Type multiplier
            node_type = data.get("type", "visit")
            type_mult = TYPE_MULTIPLIERS.get(node_type, 0.5)

            # Bonus for abnormal labs
            if data.get("is_abnormal", False):
                type_mult *= 1.5

            # 3. Graph centrality
            cent = centrality.get(node_id, 0.0)

            # Combine
            combined = (
                self.temporal_w * temporal
                + self.type_w * type_mult
                + self.centrality_w * cent
            )

            results.append(NodeImportance(
                node_id=node_id,
                description=data.get("description", ""),
                temporal_weight=round(temporal, 4),
                type_weight=round(type_mult, 4),
                centrality_weight=round(cent, 4),
                combined_score=round(combined, 4),
            ))

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

    def get_top_insights(
        self,
        graph: TemporalGraph,
        reference_date: date | None = None,
        top_k: int = 5,
    ) -> list[str]:
        """Generate human-readable temporal insights from the graph.

        Returns top-k most important events with context.
        """
        importances = self.encode(graph, reference_date)
        if not importances:
            return ["No clinical history available for temporal analysis."]

        G = graph.graph
        insights: list[str] = []

        for imp in importances[:top_k]:
            node_data = G.nodes[imp.node_id]
            node_date = node_data.get("date", date.today())
            days = node_data.get("days_ago", 0)
            node_type = node_data.get("type", "event")

            # Check if this node has causal edges
            causal_targets = [
                G.nodes[v].get("description", v)
                for _, v, d in G.out_edges(imp.node_id, data=True)
                if d.get("type") == "causal"
            ]

            time_str = f"{days} days ago" if days > 0 else "current"
            insight = f"[{node_type.upper()}] {imp.description} ({time_str}, importance: {imp.combined_score:.2f})"

            if causal_targets:
                insight += f" → linked to: {', '.join(causal_targets[:3])}"

            insights.append(insight)

        return insights

    def compute_embedding(
        self,
        graph: TemporalGraph,
        reference_date: date | None = None,
        embedding_dim: int = 64,
    ) -> list[float]:
        """Compute a fixed-size embedding vector from the temporal graph.

        Uses weighted node feature aggregation (no PyTorch required).
        The embedding captures temporal patterns, event types, and graph structure.
        """
        importances = self.encode(graph, reference_date)
        if not importances:
            return [0.0] * embedding_dim

        G = graph.graph
        # Feature extraction per node
        type_map = {"diagnosis": 0, "medication": 1, "lab": 2, "procedure": 3, "visit": 4, "imaging": 5}
        num_types = len(type_map)

        # Build weighted feature vector
        # Sections: [type_distribution (6)] + [temporal_stats (8)] + [graph_stats (8)] + [padding]
        type_dist = [0.0] * num_types
        temporal_vals: list[float] = []
        abnormal_count = 0
        total_weight = 0.0

        for imp in importances:
            data = G.nodes[imp.node_id]
            w = imp.combined_score
            total_weight += w

            # Type distribution (weighted)
            ntype = data.get("type", "visit")
            idx = type_map.get(ntype, 4)
            type_dist[idx] += w

            temporal_vals.append(imp.temporal_weight)

            if data.get("is_abnormal", False):
                abnormal_count += 1

        # Normalize type distribution
        if total_weight > 0:
            type_dist = [v / total_weight for v in type_dist]

        # Temporal statistics
        temporal_stats = [
            len(importances) / 50.0,  # Node count (normalized)
            G.number_of_edges() / 100.0,  # Edge count (normalized)
            sum(temporal_vals) / max(len(temporal_vals), 1),  # Mean temporal weight
            max(temporal_vals) if temporal_vals else 0.0,
            min(temporal_vals) if temporal_vals else 0.0,
            abnormal_count / max(len(importances), 1),  # Abnormal ratio
            sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "causal") / max(G.number_of_edges(), 1),
            importances[0].combined_score if importances else 0.0,  # Top importance
        ]

        # Graph structure stats
        try:
            avg_clustering = nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 2 else 0.0
        except Exception:
            avg_clustering = 0.0

        causal_edge_count = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "causal")

        graph_stats = [
            G.number_of_nodes() / 50.0,
            causal_edge_count / max(G.number_of_edges(), 1),
            avg_clustering,
            nx.density(G) if G.number_of_nodes() > 1 else 0.0,
            0.0, 0.0, 0.0, 0.0,  # Reserved slots
        ]

        # Assemble embedding
        raw = type_dist + temporal_stats + graph_stats
        # Pad or truncate to embedding_dim
        if len(raw) < embedding_dim:
            raw.extend([0.0] * (embedding_dim - len(raw)))
        else:
            raw = raw[:embedding_dim]

        return [round(v, 6) for v in raw]
