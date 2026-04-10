"""Convert patient clinical history into a temporal directed graph.

Each clinical event (diagnosis, medication, lab, procedure, etc.) becomes a node.
Edges encode temporal ordering (with time-decay weights) and known causal links.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import networkx as nx

from nstea.models.patient import PatientInput

# Known causal relationships: (cause_keyword, effect_keyword, relationship_label)
KNOWN_CAUSAL_LINKS: list[tuple[str, str, str]] = [
    ("nsaid", "cardiovascular", "cv_risk_increase"),
    ("nsaid", "gi bleed", "gi_risk_increase"),
    ("nsaid", "renal", "renal_risk"),
    ("diabetes", "cardiovascular", "cv_comorbidity"),
    ("diabetes", "renal", "renal_comorbidity"),
    ("hypertension", "cardiovascular", "cv_comorbidity"),
    ("hypertension", "stroke", "stroke_risk"),
    ("smoking", "copd", "respiratory_risk"),
    ("smoking", "cardiovascular", "cv_risk_increase"),
    ("obesity", "diabetes", "metabolic_risk"),
    ("atrial fibrillation", "stroke", "thromboembolic_risk"),
    ("heart failure", "renal", "cardiorenal"),
    ("ckd", "anemia", "renal_anemia"),
    ("corticosteroid", "diabetes", "steroid_diabetes"),
    ("statin", "myopathy", "drug_adverse_effect"),
    ("anticoagulant", "bleed", "bleeding_risk"),
    ("opioid", "respiratory depression", "drug_adverse_effect"),
]


@dataclass
class TemporalGraph:
    """Container for a patient's temporal clinical graph."""

    patient_id: str
    graph: nx.DiGraph
    node_count: int = 0
    edge_count: int = 0
    date_range: tuple[date | None, date | None] = (None, None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/cache storage."""
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            node_data = dict(data)
            if "date" in node_data and isinstance(node_data["date"], date):
                node_data["date"] = node_data["date"].isoformat()
            nodes.append({"id": nid, **node_data})

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, **data})

        return {
            "patient_id": self.patient_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }


class PatientGraphBuilder:
    """Converts patient clinical history into a temporal directed graph.

    Design:
    - Each clinical event → node with features (type, date, description, etc.)
    - Temporal edges: chronological ordering with exponential time-decay weights
    - Causal edges: known clinical relationships (e.g., NSAID use → CV risk)
    """

    def __init__(self, decay_lambda: float = 0.01):
        """
        Args:
            decay_lambda: Decay rate for temporal edge weights.
                Higher values = faster decay of older events.
                0.01 → events ~100 days apart have weight ~0.37
        """
        self.decay_lambda = decay_lambda

    def build_graph(self, patient: PatientInput) -> TemporalGraph:
        """Build temporal graph from patient data.

        Incorporates:
        - Clinical history events
        - Active conditions (as ongoing nodes)
        - Current medications (as ongoing nodes)
        - Recent lab results
        """
        G = nx.DiGraph()
        today = date.today()

        # --- Add nodes from clinical history ---
        for i, event in enumerate(patient.history):
            node_id = f"event_{i}"
            G.add_node(node_id, **{
                "type": event.event_type,
                "description": event.description,
                "date": event.date,
                "days_ago": (today - event.date).days,
                "category": "history",
            })

        # --- Add nodes from active conditions ---
        for i, cond in enumerate(patient.conditions):
            node_id = f"condition_{i}"
            cond_date = cond.onset_date or today
            G.add_node(node_id, **{
                "type": "diagnosis",
                "description": cond.name,
                "date": cond_date,
                "days_ago": (today - cond_date).days,
                "category": "condition",
                "status": cond.status,
                "icd10": cond.icd10_code or "",
            })

        # --- Add nodes from current medications ---
        for i, med in enumerate(patient.medications):
            node_id = f"medication_{i}"
            med_date = med.start_date or today
            G.add_node(node_id, **{
                "type": "medication",
                "description": med.name,
                "date": med_date,
                "days_ago": (today - med_date).days,
                "category": "medication",
                "dosage": med.dosage or "",
            })

        # --- Add nodes from lab results ---
        for i, lab in enumerate(patient.lab_results):
            node_id = f"lab_{i}"
            G.add_node(node_id, **{
                "type": "lab",
                "description": f"{lab.test_name}: {lab.value} {lab.unit}",
                "date": lab.date,
                "days_ago": (today - lab.date).days,
                "category": "lab",
                "is_abnormal": lab.is_abnormal or False,
                "value": lab.value,
            })

        # --- Add temporal edges (chronological ordering) ---
        sorted_nodes = sorted(
            G.nodes(data=True),
            key=lambda x: x[1].get("date", today),
        )

        for i in range(len(sorted_nodes) - 1):
            curr_id, curr_data = sorted_nodes[i]
            next_id, next_data = sorted_nodes[i + 1]

            curr_date = curr_data.get("date", today)
            next_date = next_data.get("date", today)
            time_gap_days = max((next_date - curr_date).days, 0)

            weight = math.exp(-self.decay_lambda * time_gap_days)

            G.add_edge(curr_id, next_id, **{
                "type": "temporal",
                "time_gap_days": time_gap_days,
                "weight": round(weight, 4),
            })

        # --- Add causal edges ---
        self._add_causal_edges(G)

        # Compute date range
        all_dates = [d.get("date") for _, d in G.nodes(data=True) if d.get("date")]
        date_range = (min(all_dates), max(all_dates)) if all_dates else (None, None)

        return TemporalGraph(
            patient_id=patient.patient_id,
            graph=G,
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            date_range=date_range,
        )

    def _add_causal_edges(self, G: nx.DiGraph) -> None:
        """Add known causal relationship edges between nodes."""
        nodes_list = list(G.nodes(data=True))

        for cause_kw, effect_kw, label in KNOWN_CAUSAL_LINKS:
            cause_nodes = [
                (nid, data) for nid, data in nodes_list
                if cause_kw in data.get("description", "").lower()
            ]
            effect_nodes = [
                (nid, data) for nid, data in nodes_list
                if effect_kw in data.get("description", "").lower()
            ]

            for cause_id, cause_data in cause_nodes:
                for effect_id, effect_data in effect_nodes:
                    if cause_id == effect_id:
                        continue
                    # Only add causal edge if cause precedes or coincides with effect
                    cause_date = cause_data.get("date", date.today())
                    effect_date = effect_data.get("date", date.today())
                    if cause_date <= effect_date:
                        G.add_edge(cause_id, effect_id, **{
                            "type": "causal",
                            "relationship": label,
                            "weight": 1.0,
                        })
