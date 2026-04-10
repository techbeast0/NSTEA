"""Unit tests for the RuleEngine safety checker."""

import pytest
from nstea.safety import RuleEngine, SafetyResult
from nstea.models.patient import PatientInput, Allergy, Condition, Medication


@pytest.fixture
def engine():
    return RuleEngine()


def _make_patient(**kwargs) -> PatientInput:
    defaults = {"patient_id": "test_001", "age": 50, "sex": "male"}
    defaults.update(kwargs)
    return PatientInput(**defaults)


class TestDirectAllergyMatch:
    def test_aspirin_allergy_blocks_aspirin(self, engine):
        patient = _make_patient(
            allergies=[Allergy(substance="Aspirin", reaction="Anaphylaxis", severity="severe")]
        )
        result = engine.check_proposed_drugs(patient, ["aspirin"])
        assert not result.is_safe
        assert any(v.severity == "critical" and "aspirin" in v.message.lower() for v in result.violations)

    def test_no_allergy_is_safe(self, engine):
        patient = _make_patient(allergies=[])
        result = engine.check_proposed_drugs(patient, ["aspirin", "metformin"])
        # May still have condition/interaction warnings, but no allergy critical
        allergy_violations = [v for v in result.violations if v.related_rule_type == "allergy_match"]
        assert len(allergy_violations) == 0


class TestCrossReactivity:
    def test_penicillin_allergy_blocks_amoxicillin(self, engine):
        patient = _make_patient(
            allergies=[Allergy(substance="Penicillin", reaction="Rash", severity="moderate")]
        )
        result = engine.check_proposed_drugs(patient, ["amoxicillin"])
        assert not result.is_safe
        assert any("amoxicillin" in v.blocked_drug for v in result.violations if v.blocked_drug)

    def test_penicillin_allergy_warns_cephalosporin(self, engine):
        patient = _make_patient(
            allergies=[Allergy(substance="Penicillin", reaction="Rash", severity="moderate")]
        )
        result = engine.check_proposed_drugs(patient, ["cephalexin"])
        warnings = [v for v in result.violations if v.severity == "warning"]
        assert len(warnings) > 0

    def test_sulfa_allergy_blocks_tmp_smx(self, engine):
        patient = _make_patient(
            allergies=[Allergy(substance="Sulfa", reaction="Rash", severity="moderate")]
        )
        result = engine.check_proposed_drugs(patient, ["trimethoprim-sulfamethoxazole"])
        assert not result.is_safe


class TestConditionDrugContraindications:
    def test_pregnancy_blocks_isotretinoin(self, engine):
        patient = _make_patient(
            sex="female",
            conditions=[Condition(name="Pregnancy", status="active")],
        )
        result = engine.check_proposed_drugs(patient, ["isotretinoin"])
        assert not result.is_safe
        assert any(v.severity == "critical" for v in result.violations)

    def test_asthma_blocks_propranolol(self, engine):
        patient = _make_patient(
            conditions=[Condition(name="Asthma", status="chronic")]
        )
        result = engine.check_proposed_drugs(patient, ["propranolol"])
        criticals = [v for v in result.violations if v.severity == "critical"]
        assert len(criticals) > 0


class TestDrugInteractions:
    def test_warfarin_nsaid_interaction(self, engine):
        patient = _make_patient(
            medications=[Medication(name="Warfarin", dosage="5mg daily")]
        )
        result = engine.check_proposed_drugs(patient, ["ibuprofen"])
        assert any("interaction" in v.related_rule_type.lower() for v in result.violations)

    def test_ssri_maoi_interaction(self, engine):
        patient = _make_patient(
            medications=[Medication(name="Fluoxetine", dosage="20mg daily")]
        )
        result = engine.check_proposed_drugs(patient, ["phenelzine"])
        assert not result.is_safe

    def test_opioid_benzodiazepine_interaction(self, engine):
        patient = _make_patient(
            medications=[Medication(name="Oxycodone", dosage="5mg PRN")]
        )
        result = engine.check_proposed_drugs(patient, ["diazepam"])
        assert any(v.severity == "critical" for v in result.violations)

    def test_no_interaction_when_unrelated(self, engine):
        patient = _make_patient(
            medications=[Medication(name="Metformin", dosage="500mg BID")]
        )
        result = engine.check_proposed_drugs(patient, ["amlodipine"])
        interaction_violations = [v for v in result.violations if v.related_rule_type == "drug_interaction"]
        assert len(interaction_violations) == 0


class TestEdgeCases:
    def test_empty_proposed_drugs(self, engine):
        patient = _make_patient()
        result = engine.check_proposed_drugs(patient, [])
        assert result.is_safe
        assert len(result.violations) == 0

    def test_case_insensitive_matching(self, engine):
        patient = _make_patient(
            allergies=[Allergy(substance="PENICILLIN", reaction="Rash", severity="severe")]
        )
        result = engine.check_proposed_drugs(patient, ["penicillin"])
        assert not result.is_safe

    def test_multiple_violations(self, engine):
        patient = _make_patient(
            allergies=[Allergy(substance="Aspirin", reaction="Anaphylaxis", severity="severe")],
            medications=[Medication(name="Warfarin", dosage="5mg daily")],
        )
        result = engine.check_proposed_drugs(patient, ["aspirin", "ibuprofen"])
        assert not result.is_safe
        assert len(result.violations) >= 2
