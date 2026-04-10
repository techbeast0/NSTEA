"""Tests for Phase 5: Symbolic Constraint Engine and Knowledge Graph."""

import pytest

from nstea.models.patient import Allergy, Condition, Medication, PatientInput
from nstea.safety.constraint_engine import ConstraintResult, SymbolicConstraintEngine
from nstea.safety.knowledge_graph import (
    ClinicalKnowledgeGraph,
    KGEdge,
    KGNode,
    build_default_knowledge_graph,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_patient(**kwargs) -> PatientInput:
    """Build a test patient with overrides."""
    defaults = {
        "patient_id": "TEST-P1",
        "age": 65,
        "sex": "male",
        "conditions": [],
        "medications": [],
        "allergies": [],
    }
    defaults.update(kwargs)
    return PatientInput(**defaults)


# ── Knowledge Graph Tests ────────────────────────────────────────────────────

class TestClinicalKnowledgeGraph:
    def test_build_default_kg(self):
        kg = build_default_knowledge_graph()
        assert kg.node_count > 30
        assert kg.edge_count > 20

    def test_find_node_exact(self):
        kg = build_default_knowledge_graph()
        node_id = kg.find_node("Aspirin")
        assert node_id == "drug:aspirin"

    def test_find_node_case_insensitive(self):
        kg = build_default_knowledge_graph()
        assert kg.find_node("ASPIRIN") is not None
        assert kg.find_node("aspirin") is not None

    def test_find_node_substring(self):
        kg = build_default_knowledge_graph()
        assert kg.find_node("Metformin") is not None

    def test_find_node_not_found(self):
        kg = build_default_knowledge_graph()
        assert kg.find_node("NonexistentDrug12345") is None

    def test_get_drug_interactions(self):
        kg = build_default_knowledge_graph()
        interactions = kg.get_drug_interactions("Warfarin")
        assert len(interactions) >= 2  # Aspirin, Ibuprofen, Omeprazole
        drugs = [i["drug"] for i in interactions]
        assert any("Aspirin" in d for d in drugs)

    def test_get_contraindications(self):
        kg = build_default_knowledge_graph()
        contras = kg.get_contraindications("Metformin")
        assert len(contras) >= 1
        conditions = [c["condition"] for c in contras]
        assert any("Kidney" in c or "CKD" in c for c in conditions)

    def test_get_allergy_cross_reactions(self):
        kg = build_default_knowledge_graph()
        reactions = kg.get_allergy_cross_reactions("Aspirin Allergy")
        assert len(reactions) >= 2  # Ibuprofen, Naproxen, Celecoxib
        drugs = [r["drug"] for r in reactions]
        assert any("Ibuprofen" in d for d in drugs)

    def test_get_guideline_recommendations(self):
        kg = build_default_knowledge_graph()
        recs = kg.get_guideline_recommendations("Myocardial Infarction")
        assert len(recs) >= 1
        drugs = [r["drug"] for r in recs]
        assert any("Aspirin" in d or "Clopidogrel" in d for d in drugs)

    def test_is_mapped(self):
        kg = build_default_knowledge_graph()
        assert kg.is_mapped("Aspirin") is True
        assert kg.is_mapped("FakeDrug999") is False

    def test_custom_node_and_edge(self):
        kg = ClinicalKnowledgeGraph()
        kg.add_node(KGNode("drug:test", "drug", "TestDrug"))
        kg.add_node(KGNode("cond:test", "condition", "TestCondition"))
        kg.add_edge(KGEdge("drug:test", "cond:test", "contraindicated_in",
                           {"severity": "critical", "reason": "test"}))
        assert kg.node_count == 2
        assert kg.edge_count == 1
        contras = kg.get_contraindications("TestDrug")
        assert len(contras) == 1


# ── Symbolic Constraint Engine Tests ─────────────────────────────────────────

class TestSymbolicConstraintEngine:
    def test_safe_prescription(self):
        """No violations for safe prescriptions."""
        patient = _make_patient(
            conditions=[Condition(name="Hypertension")],
            medications=[Medication(name="Amlodipine")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Lisinopril"], patient)
        # No critical violations
        critical = [v for v in result.violations if v.severity == "critical"]
        assert len(critical) == 0

    def test_allergy_direct_match(self):
        """Direct allergy match should produce critical violation."""
        patient = _make_patient(
            allergies=[Allergy(substance="Aspirin", severity="severe")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Aspirin"], patient)
        assert not result.is_safe
        assert any(v.rule == "ALLERGY_DIRECT_MATCH" for v in result.violations)

    def test_allergy_cross_reactivity(self):
        """Aspirin allergy should flag NSAIDs as cross-reactive."""
        patient = _make_patient(
            allergies=[Allergy(substance="Aspirin", severity="severe")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Ibuprofen"], patient)
        assert any(v.rule == "ALLERGY_CROSS_REACTIVITY" for v in result.violations)

    def test_drug_interaction(self):
        """Warfarin + Aspirin should flag critical interaction."""
        patient = _make_patient(
            medications=[Medication(name="Warfarin")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Aspirin"], patient)
        assert any(v.rule == "DRUG_INTERACTION" for v in result.violations)
        assert any(v.severity == "critical" for v in result.violations)

    def test_condition_contraindication(self):
        """Metformin should be flagged with CKD."""
        patient = _make_patient(
            conditions=[Condition(name="Chronic Kidney Disease")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Metformin"], patient)
        assert not result.is_safe
        assert any(v.rule == "CONDITION_CONTRAINDICATION" for v in result.violations)

    def test_beta_blocker_asthma(self):
        """Beta-blockers contraindicated in asthma."""
        patient = _make_patient(
            conditions=[Condition(name="Asthma")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Metoprolol"], patient)
        assert not result.is_safe

    def test_unmapped_drug_flagged(self):
        """Drugs not in KG should be flagged as unmapped."""
        patient = _make_patient()
        engine = SymbolicConstraintEngine()
        result = engine.validate(["SomeExperimentalDrug"], patient)
        assert "SomeExperimentalDrug" in result.unmapped_actions
        assert result.requires_human_review

    def test_guideline_alignment(self):
        """Check that guideline recommendations are reported."""
        patient = _make_patient(
            conditions=[Condition(name="Myocardial Infarction")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Aspirin", "Clopidogrel"], patient)
        assert len(result.guideline_alignment) > 0
        # Check that proposed drugs are marked as aligned
        aligned = [g for g in result.guideline_alignment if g["is_proposed"]]
        assert len(aligned) > 0

    def test_multiple_violations(self):
        """Multiple violations for a complex case."""
        patient = _make_patient(
            conditions=[Condition(name="Chronic Kidney Disease"), Condition(name="Asthma")],
            medications=[Medication(name="Warfarin")],
            allergies=[Allergy(substance="Penicillin", severity="severe")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Metformin", "Metoprolol", "Amoxicillin", "Aspirin"], patient)
        assert len(result.violations) >= 3  # CKD+metformin, asthma+metoprolol, penicillin+amoxicillin, warfarin+aspirin

    def test_constraint_result_properties(self):
        """Test ConstraintResult properties."""
        patient = _make_patient(
            allergies=[Allergy(substance="Aspirin", severity="severe")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Aspirin", "UnknownDrug123"], patient)
        assert result.has_critical
        assert result.requires_human_review
        assert "critical" in result.review_reason.lower() or "unmapped" in result.review_reason.lower()

    def test_empty_proposed_drugs(self):
        """No violations when no drugs proposed."""
        patient = _make_patient()
        engine = SymbolicConstraintEngine()
        result = engine.validate([], patient)
        assert result.is_safe
        assert len(result.violations) == 0

    def test_serotonin_syndrome_risk(self):
        """SSRI + Tramadol → serotonin syndrome."""
        patient = _make_patient(
            medications=[Medication(name="SSRI")],
        )
        engine = SymbolicConstraintEngine()
        result = engine.validate(["Tramadol"], patient)
        assert any("serotonin" in v.description.lower() for v in result.violations)
