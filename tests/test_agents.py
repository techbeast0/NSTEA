"""Tests for the safety and confidence agents."""

import pytest
from nstea.agents.safety_agent import run_safety_check
from nstea.agents.confidence_agent import evaluate_confidence
from nstea.models.patient import PatientInput, Allergy, Condition, Medication
from nstea.models.analysis import (
    AnalysisResponse,
    ConfidenceScore,
    DiagnosisOutput,
    Recommendation,
    ReasoningStep,
    SafetyFlag,
)


def _make_patient(**kwargs) -> PatientInput:
    defaults = {"patient_id": "test_001", "age": 50, "sex": "male"}
    defaults.update(kwargs)
    return PatientInput(**defaults)


def _make_analysis(**kwargs) -> AnalysisResponse:
    defaults = {
        "analysis_id": "test_001",
        "patient_id": "test_001",
        "diagnosis": DiagnosisOutput(primary="Test Diagnosis"),
        "recommendations": [
            Recommendation(
                action="Prescribe metformin 500mg BID",
                category="medication",
                urgency="routine",
                rationale="Blood glucose control",
            )
        ],
        "reasoning_steps": [
            ReasoningStep(step_number=1, description="Analysis", input_summary="data", output_summary="result"),
            ReasoningStep(step_number=2, description="Synthesis", input_summary="data2", output_summary="result2"),
        ],
        "confidence": ConfidenceScore(overall=0.8, evidence_strength=0.7, model_certainty=0.8),
    }
    defaults.update(kwargs)
    return AnalysisResponse(**defaults)


class TestSafetyAgent:
    def test_safe_recommendation_passes(self):
        patient = _make_patient()
        analysis = _make_analysis()
        result = run_safety_check(patient, analysis)
        # No allergies, no interactions — should pass clean
        critical_flags = [f for f in result.safety_flags if f.level == "critical" and "RULE ENGINE" in f.message]
        assert len(critical_flags) == 0

    def test_allergy_violation_flagged(self):
        patient = _make_patient(
            allergies=[Allergy(substance="Aspirin", reaction="Anaphylaxis", severity="severe")]
        )
        analysis = _make_analysis(
            recommendations=[
                Recommendation(
                    action="aspirin 81mg daily",
                    category="medication",
                    urgency="routine",
                    rationale="Antiplatelet therapy",
                )
            ]
        )
        result = run_safety_check(patient, analysis)
        assert result.requires_human_review
        assert any("RULE ENGINE" in f.message for f in result.safety_flags)
        # Critical violation should remove the recommendation
        assert len(result.recommendations) == 0
        assert any("REMOVED" in f.message for f in result.safety_flags)

    def test_interaction_flagged(self):
        patient = _make_patient(
            medications=[Medication(name="Warfarin", dosage="5mg daily")]
        )
        analysis = _make_analysis(
            recommendations=[
                Recommendation(
                    action="ibuprofen 400mg TID",
                    category="medication",
                    urgency="routine",
                    rationale="Pain management",
                )
            ]
        )
        result = run_safety_check(patient, analysis)
        assert any("RULE ENGINE" in f.message for f in result.safety_flags)
        # Critical interaction should remove the recommendation
        assert len(result.recommendations) == 0

    def test_no_medication_recs_passes(self):
        patient = _make_patient()
        analysis = _make_analysis(
            recommendations=[
                Recommendation(
                    action="Order CBC and BMP",
                    category="test",
                    urgency="routine",
                    rationale="Baseline labs",
                )
            ]
        )
        result = run_safety_check(patient, analysis)
        # No meds to check — should return unchanged
        rule_flags = [f for f in result.safety_flags if "RULE ENGINE" in f.message]
        assert len(rule_flags) == 0


class TestConfidenceAgent:
    def test_high_confidence_passes(self):
        analysis = _make_analysis(
            confidence=ConfidenceScore(overall=0.85, evidence_strength=0.8, model_certainty=0.9)
        )
        result = evaluate_confidence(analysis)
        assert not result.requires_human_review

    def test_low_confidence_escalates(self):
        analysis = _make_analysis(
            confidence=ConfidenceScore(overall=0.3, evidence_strength=0.2, model_certainty=0.4)
        )
        result = evaluate_confidence(analysis)
        assert result.requires_human_review
        assert "Low confidence" in result.escalation_reason

    def test_no_reasoning_escalates(self):
        analysis = _make_analysis(reasoning_steps=[])
        result = evaluate_confidence(analysis)
        assert result.requires_human_review
        assert "reasoning" in result.escalation_reason.lower()

    def test_no_recommendations_escalates(self):
        analysis = _make_analysis(recommendations=[])
        result = evaluate_confidence(analysis)
        assert result.requires_human_review
        assert "recommendation" in result.escalation_reason.lower()

    def test_critical_safety_flag_escalates(self):
        analysis = _make_analysis(
            safety_flags=[
                SafetyFlag(level="critical", message="Allergy violation detected")
            ]
        )
        result = evaluate_confidence(analysis)
        assert result.requires_human_review
        assert "critical safety" in result.escalation_reason.lower()

    def test_fallback_diagnosis_escalates(self):
        analysis = _make_analysis(
            diagnosis=DiagnosisOutput(primary="FREE-TEXT RESPONSE (no JSON)")
        )
        result = evaluate_confidence(analysis)
        assert result.requires_human_review

    def test_existing_escalation_preserved(self):
        analysis = _make_analysis(
            escalation_reason="Prior issue",
            confidence=ConfidenceScore(overall=0.3, evidence_strength=0.2, model_certainty=0.4),
        )
        result = evaluate_confidence(analysis)
        assert "Prior issue" in result.escalation_reason
        assert "CONFIDENCE" in result.escalation_reason
