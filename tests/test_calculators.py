"""Tests for clinical calculators."""

import pytest

from nstea.tools.lab_calculator import (
    calc_cha2ds2_vasc,
    calc_curb65,
    calc_egfr,
    calc_meld,
    calc_wells_dvt,
)


# ---------------------------------------------------------------------------
# eGFR
# ---------------------------------------------------------------------------

class TestEGFR:
    def test_normal_male(self):
        result = calc_egfr(creatinine=0.9, age=30, sex="male")
        assert result.score > 90
        assert "Normal" in result.interpretation

    def test_normal_female(self):
        result = calc_egfr(creatinine=0.7, age=25, sex="female")
        assert result.score > 90

    def test_reduced_elderly(self):
        result = calc_egfr(creatinine=1.8, age=75, sex="male")
        assert result.score < 60
        assert "G3" in result.interpretation or "G4" in result.interpretation or "G5" in result.interpretation

    def test_severe_reduction(self):
        result = calc_egfr(creatinine=5.0, age=60, sex="male")
        assert result.score < 20


# ---------------------------------------------------------------------------
# CHA2DS2-VASc
# ---------------------------------------------------------------------------

class TestCHA2DS2VASc:
    def test_zero_score(self):
        result = calc_cha2ds2_vasc(age=50, sex="male")
        assert result.score == 0
        assert "Low risk" in result.interpretation

    def test_max_score_components(self):
        result = calc_cha2ds2_vasc(
            age=76, sex="female",
            has_chf=True, has_hypertension=True,
            has_stroke_tia=True, has_vascular_disease=True,
            has_diabetes=True,
        )
        assert result.score == 9

    def test_age_65_74(self):
        result = calc_cha2ds2_vasc(age=70, sex="male")
        assert result.score == 1

    def test_female_adds_point(self):
        male = calc_cha2ds2_vasc(age=50, sex="male")
        female = calc_cha2ds2_vasc(age=50, sex="female")
        assert female.score == male.score + 1


# ---------------------------------------------------------------------------
# MELD
# ---------------------------------------------------------------------------

class TestMELD:
    def test_low_severity(self):
        result = calc_meld(bilirubin=1.0, inr=1.0, creatinine=0.8)
        assert result.score <= 10
        assert "Low" in result.interpretation

    def test_high_severity(self):
        result = calc_meld(bilirubin=10.0, inr=3.0, creatinine=3.5, sodium=125)
        assert result.score >= 30

    def test_dialysis_sets_creatinine(self):
        result = calc_meld(bilirubin=2.0, inr=1.5, creatinine=1.0, on_dialysis=True)
        assert result.details["on_dialysis"] is True

    def test_score_clamped(self):
        result = calc_meld(bilirubin=0.5, inr=0.8, creatinine=0.5)
        assert result.score >= 6


# ---------------------------------------------------------------------------
# Wells DVT
# ---------------------------------------------------------------------------

class TestWellsDVT:
    def test_low_probability(self):
        result = calc_wells_dvt()
        assert result.score == 0
        assert "Low" in result.interpretation

    def test_high_probability(self):
        result = calc_wells_dvt(
            active_cancer=True,
            tenderness_along_veins=True,
            entire_leg_swollen=True,
            previous_dvt=True,
        )
        assert result.score >= 3
        assert "High" in result.interpretation

    def test_alternative_diagnosis_subtracts(self):
        result = calc_wells_dvt(active_cancer=True, alternative_diagnosis_likely=True)
        assert result.score == -1


# ---------------------------------------------------------------------------
# CURB-65
# ---------------------------------------------------------------------------

class TestCURB65:
    def test_low_severity(self):
        result = calc_curb65()
        assert result.score == 0
        assert "Low" in result.interpretation

    def test_severe(self):
        result = calc_curb65(
            confusion=True,
            bun_gt_19=True,
            respiratory_rate_ge_30=True,
            systolic_bp_lt_90=True,
            age_ge_65=True,
        )
        assert result.score == 5
        assert "ICU" in result.interpretation

    def test_moderate(self):
        result = calc_curb65(confusion=True, age_ge_65=True)
        assert result.score == 2
        assert "Moderate" in result.interpretation
