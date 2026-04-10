"""In-memory knowledge graph for clinical constraints.

Uses NetworkX as the backbone. In production, swap for Neo4j backend.
Stores drug interactions, contraindications, allergy cross-reactivities,
and guideline recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

import structlog

logger = structlog.stdlib.get_logger(__name__)


@dataclass
class KGNode:
    """A node in the clinical knowledge graph."""

    id: str
    type: Literal["drug", "condition", "allergy", "guideline"]
    name: str
    codes: dict[str, str] = field(default_factory=dict)  # e.g., {"rxnorm": "1234", "icd10": "E11"}
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGEdge:
    """A relationship in the clinical knowledge graph."""

    source_id: str
    target_id: str
    type: Literal[
        "interacts_with",
        "contraindicated_in",
        "causes_allergy",
        "cross_reacts_with",
        "recommends",
        "for_condition",
        "requires_caution",
    ]
    properties: dict[str, Any] = field(default_factory=dict)


class ClinicalKnowledgeGraph:
    """In-memory clinical knowledge graph backed by NetworkX.

    Node types: drug, condition, allergy, guideline
    Edge types: interacts_with, contraindicated_in, causes_allergy,
                cross_reacts_with, recommends, for_condition, requires_caution

    Provides query methods used by the SymbolicConstraintEngine.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_index: dict[str, str] = {}  # lowercase_name -> node_id

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def add_node(self, node: KGNode) -> None:
        """Add a node to the knowledge graph."""
        self.graph.add_node(node.id, **{
            "type": node.type,
            "name": node.name,
            "codes": node.codes,
            **node.properties,
        })
        self._node_index[node.name.lower()] = node.id

    def add_edge(self, edge: KGEdge) -> None:
        """Add a relationship to the knowledge graph."""
        self.graph.add_edge(edge.source_id, edge.target_id, **{
            "type": edge.type,
            **edge.properties,
        })

    def find_node(self, name: str) -> str | None:
        """Find a node by name (case-insensitive, fuzzy substring match)."""
        name_lower = name.lower().strip()
        # Exact match first
        if name_lower in self._node_index:
            return self._node_index[name_lower]
        # Substring match
        for indexed_name, node_id in self._node_index.items():
            if name_lower in indexed_name or indexed_name in name_lower:
                return node_id
        return None

    def get_drug_interactions(self, drug_name: str) -> list[dict[str, Any]]:
        """Get all drugs that interact with the given drug."""
        node_id = self.find_node(drug_name)
        if node_id is None:
            return []

        interactions = []
        # Check outgoing edges
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("type") == "interacts_with":
                target_data = self.graph.nodes.get(target, {})
                interactions.append({
                    "drug": target_data.get("name", target),
                    "severity": data.get("severity", "unknown"),
                    "effect": data.get("effect", ""),
                    "mechanism": data.get("mechanism", ""),
                })
        # Check incoming edges (interactions are bidirectional)
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if data.get("type") == "interacts_with":
                source_data = self.graph.nodes.get(source, {})
                interactions.append({
                    "drug": source_data.get("name", source),
                    "severity": data.get("severity", "unknown"),
                    "effect": data.get("effect", ""),
                    "mechanism": data.get("mechanism", ""),
                })
        return interactions

    def get_contraindications(self, drug_name: str) -> list[dict[str, Any]]:
        """Get conditions where the drug is contraindicated."""
        node_id = self.find_node(drug_name)
        if node_id is None:
            return []

        results = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("type") == "contraindicated_in":
                target_data = self.graph.nodes.get(target, {})
                results.append({
                    "condition": target_data.get("name", target),
                    "severity": data.get("severity", "critical"),
                    "reason": data.get("reason", ""),
                })
        return results

    def get_allergy_cross_reactions(self, allergy_name: str) -> list[dict[str, Any]]:
        """Get drugs that cross-react with the given allergy."""
        # Try finding an allergy-type node specifically (allergy nodes are named "X Allergy")
        node_id = self.find_node(f"{allergy_name} Allergy")
        if node_id is None or self.graph.nodes.get(node_id, {}).get("type") != "allergy":
            node_id = self.find_node(allergy_name)
        if node_id is None:
            return []

        results = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("type") == "cross_reacts_with":
                target_data = self.graph.nodes.get(target, {})
                results.append({
                    "drug": target_data.get("name", target),
                    "risk_level": data.get("risk_level", "high"),
                    "alternative": data.get("alternative", ""),
                })
        return results

    def get_guideline_recommendations(self, condition_name: str) -> list[dict[str, Any]]:
        """Get guideline-recommended drugs for a condition."""
        node_id = self.find_node(condition_name)
        if node_id is None:
            return []

        results = []
        # Find guidelines linked to this condition
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if data.get("type") == "for_condition":
                source_data = self.graph.nodes.get(source, {})
                if source_data.get("type") == "guideline":
                    # Get drugs recommended by this guideline
                    for _, drug_id, rec_data in self.graph.out_edges(source, data=True):
                        if rec_data.get("type") == "recommends":
                            drug_data = self.graph.nodes.get(drug_id, {})
                            results.append({
                                "drug": drug_data.get("name", drug_id),
                                "guideline": source_data.get("name", source),
                                "strength": rec_data.get("strength", "moderate"),
                            })
        return results

    def is_mapped(self, name: str) -> bool:
        """Check if a drug/condition exists in the knowledge graph."""
        return self.find_node(name) is not None


def build_default_knowledge_graph() -> ClinicalKnowledgeGraph:
    """Build the default clinical knowledge graph with common rules.

    This replaces and extends the YAML rule engine data with structured graph data.
    """
    kg = ClinicalKnowledgeGraph()

    # ── DRUGS ────────────────────────────────────────────────────────────
    drugs = [
        KGNode("drug:aspirin", "drug", "Aspirin", {"rxnorm": "1191"}),
        KGNode("drug:ibuprofen", "drug", "Ibuprofen", {"rxnorm": "5640"}),
        KGNode("drug:naproxen", "drug", "Naproxen", {"rxnorm": "7258"}),
        KGNode("drug:celecoxib", "drug", "Celecoxib", {"rxnorm": "140587"}),
        KGNode("drug:clopidogrel", "drug", "Clopidogrel", {"rxnorm": "32968"}),
        KGNode("drug:warfarin", "drug", "Warfarin", {"rxnorm": "11289"}),
        KGNode("drug:heparin", "drug", "Heparin", {"rxnorm": "5224"}),
        KGNode("drug:metformin", "drug", "Metformin", {"rxnorm": "6809"}),
        KGNode("drug:lisinopril", "drug", "Lisinopril", {"rxnorm": "29046"}),
        KGNode("drug:enalapril", "drug", "Enalapril", {"rxnorm": "3827"}),
        KGNode("drug:losartan", "drug", "Losartan", {"rxnorm": "52175"}),
        KGNode("drug:metoprolol", "drug", "Metoprolol", {"rxnorm": "6918"}),
        KGNode("drug:atenolol", "drug", "Atenolol", {"rxnorm": "1202"}),
        KGNode("drug:amlodipine", "drug", "Amlodipine", {"rxnorm": "17767"}),
        KGNode("drug:simvastatin", "drug", "Simvastatin", {"rxnorm": "36567"}),
        KGNode("drug:atorvastatin", "drug", "Atorvastatin", {"rxnorm": "83367"}),
        KGNode("drug:omeprazole", "drug", "Omeprazole", {"rxnorm": "7646"}),
        KGNode("drug:amoxicillin", "drug", "Amoxicillin", {"rxnorm": "723"}),
        KGNode("drug:penicillin", "drug", "Penicillin", {"rxnorm": "7980"}),
        KGNode("drug:cephalexin", "drug", "Cephalexin", {"rxnorm": "2231"}),
        KGNode("drug:ciprofloxacin", "drug", "Ciprofloxacin", {"rxnorm": "2551"}),
        KGNode("drug:prednisone", "drug", "Prednisone", {"rxnorm": "8640"}),
        KGNode("drug:insulin", "drug", "Insulin", {"rxnorm": "5856"}),
        KGNode("drug:furosemide", "drug", "Furosemide", {"rxnorm": "4603"}),
        KGNode("drug:spironolactone", "drug", "Spironolactone", {"rxnorm": "9997"}),
        KGNode("drug:digoxin", "drug", "Digoxin", {"rxnorm": "3407"}),
        KGNode("drug:lithium", "drug", "Lithium", {"rxnorm": "6448"}),
        KGNode("drug:phenytoin", "drug", "Phenytoin", {"rxnorm": "8183"}),
        KGNode("drug:tramadol", "drug", "Tramadol", {"rxnorm": "10689"}),
        KGNode("drug:ssri", "drug", "SSRI", {"rxnorm": ""}),
    ]

    # ── CONDITIONS ───────────────────────────────────────────────────────
    conditions = [
        KGNode("cond:diabetes_t2", "condition", "Type 2 Diabetes", {"icd10": "E11.9"}),
        KGNode("cond:hypertension", "condition", "Hypertension", {"icd10": "I10"}),
        KGNode("cond:heart_failure", "condition", "Heart Failure", {"icd10": "I50.9"}),
        KGNode("cond:ckd", "condition", "Chronic Kidney Disease", {"icd10": "N18.9"}),
        KGNode("cond:asthma", "condition", "Asthma", {"icd10": "J45"}),
        KGNode("cond:copd", "condition", "COPD", {"icd10": "J44.1"}),
        KGNode("cond:afib", "condition", "Atrial Fibrillation", {"icd10": "I48.91"}),
        KGNode("cond:gi_bleed", "condition", "GI Bleeding", {"icd10": "K92.2"}),
        KGNode("cond:liver_disease", "condition", "Liver Disease", {"icd10": "K76.9"}),
        KGNode("cond:gout", "condition", "Gout", {"icd10": "M10.9"}),
        KGNode("cond:peptic_ulcer", "condition", "Peptic Ulcer Disease", {"icd10": "K27.9"}),
        KGNode("cond:mi", "condition", "Myocardial Infarction", {"icd10": "I21.9"}),
        KGNode("cond:stroke", "condition", "Stroke", {"icd10": "I63.9"}),
    ]

    # ── ALLERGIES ────────────────────────────────────────────────────────
    allergies = [
        KGNode("allergy:aspirin", "allergy", "Aspirin Allergy"),
        KGNode("allergy:nsaid", "allergy", "NSAID Allergy"),
        KGNode("allergy:penicillin", "allergy", "Penicillin Allergy"),
        KGNode("allergy:sulfa", "allergy", "Sulfa Allergy"),
        KGNode("allergy:ace_inhibitor", "allergy", "ACE Inhibitor Allergy"),
    ]

    # ── GUIDELINES ───────────────────────────────────────────────────────
    guidelines = [
        KGNode("gl:acc_aha_acs_2023", "guideline", "ACC/AHA 2023 ACS Guidelines"),
        KGNode("gl:esc_hf_2021", "guideline", "ESC 2021 Heart Failure Guidelines"),
        KGNode("gl:kdigo_ckd_2024", "guideline", "KDIGO 2024 CKD Guidelines"),
        KGNode("gl:ada_diabetes_2024", "guideline", "ADA 2024 Diabetes Guidelines"),
        KGNode("gl:gina_asthma_2024", "guideline", "GINA 2024 Asthma Guidelines"),
    ]

    for node in drugs + conditions + allergies + guidelines:
        kg.add_node(node)

    # ── DRUG INTERACTIONS ────────────────────────────────────────────────
    interactions = [
        # Warfarin interactions
        KGEdge("drug:warfarin", "drug:aspirin", "interacts_with",
               {"severity": "critical", "effect": "Increased bleeding risk",
                "mechanism": "Additive anticoagulant + antiplatelet effects"}),
        KGEdge("drug:warfarin", "drug:ibuprofen", "interacts_with",
               {"severity": "critical", "effect": "Increased bleeding risk",
                "mechanism": "NSAIDs inhibit platelet function + GI erosion"}),
        KGEdge("drug:warfarin", "drug:omeprazole", "interacts_with",
               {"severity": "warning", "effect": "Altered warfarin metabolism",
                "mechanism": "CYP2C19 inhibition may increase warfarin levels"}),
        # ACE inhibitor + potassium-sparing diuretic → hyperkalemia
        KGEdge("drug:lisinopril", "drug:spironolactone", "interacts_with",
               {"severity": "warning", "effect": "Hyperkalemia risk",
                "mechanism": "Both increase potassium retention"}),
        # Metformin + contrast dye → lactic acidosis (simplified)
        KGEdge("drug:metformin", "drug:furosemide", "interacts_with",
               {"severity": "warning", "effect": "Potential lactic acidosis with renal impairment",
                "mechanism": "Furosemide may impair renal function affecting metformin clearance"}),
        # SSRI + Tramadol → serotonin syndrome
        KGEdge("drug:ssri", "drug:tramadol", "interacts_with",
               {"severity": "critical", "effect": "Serotonin syndrome risk",
                "mechanism": "Both increase serotonergic activity"}),
        # Digoxin + Furosemide → hypokalemia → digoxin toxicity
        KGEdge("drug:digoxin", "drug:furosemide", "interacts_with",
               {"severity": "warning", "effect": "Digoxin toxicity risk via hypokalemia",
                "mechanism": "Loop diuretics cause potassium loss, increasing digoxin sensitivity"}),
        # Clopidogrel + Omeprazole → reduced clopidogrel effect
        KGEdge("drug:clopidogrel", "drug:omeprazole", "interacts_with",
               {"severity": "warning", "effect": "Reduced antiplatelet effect",
                "mechanism": "Omeprazole inhibits CYP2C19, reducing clopidogrel activation"}),
        # Simvastatin + amlodipine → increased statin toxicity
        KGEdge("drug:simvastatin", "drug:amlodipine", "interacts_with",
               {"severity": "warning", "effect": "Increased risk of myopathy/rhabdomyolysis",
                "mechanism": "Amlodipine inhibits CYP3A4, increasing simvastatin levels"}),
    ]

    # ── CONTRAINDICATIONS ────────────────────────────────────────────────
    contraindications = [
        KGEdge("drug:metformin", "cond:ckd", "contraindicated_in",
               {"severity": "critical", "reason": "Lactic acidosis risk with eGFR <30"}),
        KGEdge("drug:ibuprofen", "cond:ckd", "contraindicated_in",
               {"severity": "critical", "reason": "NSAID nephrotoxicity"}),
        KGEdge("drug:ibuprofen", "cond:gi_bleed", "contraindicated_in",
               {"severity": "critical", "reason": "NSAIDs worsen GI bleeding"}),
        KGEdge("drug:ibuprofen", "cond:peptic_ulcer", "contraindicated_in",
               {"severity": "critical", "reason": "NSAIDs worsen peptic ulcer disease"}),
        KGEdge("drug:aspirin", "cond:gi_bleed", "contraindicated_in",
               {"severity": "critical", "reason": "Aspirin worsens GI bleeding"}),
        KGEdge("drug:metoprolol", "cond:asthma", "contraindicated_in",
               {"severity": "critical", "reason": "Beta-blockers cause bronchospasm in asthma"}),
        KGEdge("drug:atenolol", "cond:asthma", "contraindicated_in",
               {"severity": "critical", "reason": "Beta-blockers cause bronchospasm in asthma"}),
        KGEdge("drug:lisinopril", "cond:ckd", "requires_caution",
               {"severity": "warning", "reason": "Monitor potassium and renal function closely"}),
        KGEdge("drug:spironolactone", "cond:ckd", "requires_caution",
               {"severity": "warning", "reason": "Hyperkalemia risk in renal impairment"}),
        KGEdge("drug:lithium", "cond:ckd", "contraindicated_in",
               {"severity": "critical", "reason": "Lithium toxicity with impaired renal clearance"}),
        KGEdge("drug:phenytoin", "cond:liver_disease", "requires_caution",
               {"severity": "warning", "reason": "Altered phenytoin metabolism in liver disease"}),
    ]

    # ── ALLERGY CROSS-REACTIVITIES ──────────────────────────────────────
    cross_reactions = [
        # Aspirin allergy → cross-reacts with NSAIDs
        KGEdge("allergy:aspirin", "drug:ibuprofen", "cross_reacts_with",
               {"risk_level": "high", "alternative": "Acetaminophen"}),
        KGEdge("allergy:aspirin", "drug:naproxen", "cross_reacts_with",
               {"risk_level": "high", "alternative": "Acetaminophen"}),
        KGEdge("allergy:aspirin", "drug:celecoxib", "cross_reacts_with",
               {"risk_level": "moderate", "alternative": "Acetaminophen"}),
        # NSAID allergy → cross-reacts with aspirin
        KGEdge("allergy:nsaid", "drug:aspirin", "cross_reacts_with",
               {"risk_level": "high", "alternative": "Acetaminophen"}),
        # Penicillin allergy → cross-reacts with some cephalosporins
        KGEdge("allergy:penicillin", "drug:cephalexin", "cross_reacts_with",
               {"risk_level": "moderate", "alternative": "Azithromycin"}),
        KGEdge("allergy:penicillin", "drug:amoxicillin", "cross_reacts_with",
               {"risk_level": "high", "alternative": "Azithromycin"}),
        # ACE inhibitor allergy
        KGEdge("allergy:ace_inhibitor", "drug:lisinopril", "cross_reacts_with",
               {"risk_level": "high", "alternative": "Losartan (ARB)"}),
        KGEdge("allergy:ace_inhibitor", "drug:enalapril", "cross_reacts_with",
               {"risk_level": "high", "alternative": "Losartan (ARB)"}),
    ]

    # ── GUIDELINE RECOMMENDATIONS ────────────────────────────────────────
    guideline_edges = [
        # ACS Guidelines
        KGEdge("gl:acc_aha_acs_2023", "cond:mi", "for_condition", {}),
        KGEdge("gl:acc_aha_acs_2023", "drug:aspirin", "recommends",
               {"strength": "strong"}),
        KGEdge("gl:acc_aha_acs_2023", "drug:clopidogrel", "recommends",
               {"strength": "strong"}),
        KGEdge("gl:acc_aha_acs_2023", "drug:atorvastatin", "recommends",
               {"strength": "strong"}),
        KGEdge("gl:acc_aha_acs_2023", "drug:metoprolol", "recommends",
               {"strength": "moderate"}),
        # Heart Failure Guidelines
        KGEdge("gl:esc_hf_2021", "cond:heart_failure", "for_condition", {}),
        KGEdge("gl:esc_hf_2021", "drug:lisinopril", "recommends",
               {"strength": "strong"}),
        KGEdge("gl:esc_hf_2021", "drug:metoprolol", "recommends",
               {"strength": "strong"}),
        KGEdge("gl:esc_hf_2021", "drug:spironolactone", "recommends",
               {"strength": "strong"}),
        # Diabetes Guidelines
        KGEdge("gl:ada_diabetes_2024", "cond:diabetes_t2", "for_condition", {}),
        KGEdge("gl:ada_diabetes_2024", "drug:metformin", "recommends",
               {"strength": "strong"}),
        KGEdge("gl:ada_diabetes_2024", "drug:insulin", "recommends",
               {"strength": "moderate"}),
    ]

    for edge in interactions + contraindications + cross_reactions + guideline_edges:
        kg.add_edge(edge)

    logger.info(
        "knowledge_graph_built",
        nodes=kg.node_count,
        edges=kg.edge_count,
    )

    return kg
