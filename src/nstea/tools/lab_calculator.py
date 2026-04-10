"""Clinical risk calculators for NS-TEA.

Provides validated implementations of common clinical scores:
- eGFR (CKD-EPI 2021)
- CHA₂DS₂-VASc (stroke risk in atrial fibrillation)
- MELD (liver disease severity)
- Wells Score (DVT probability)
- CURB-65 (pneumonia severity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalculatorResult:
    """Standardized output for all calculators."""
    calculator: str
    score: float
    interpretation: str
    details: dict


# ---------------------------------------------------------------------------
# eGFR — CKD-EPI 2021 (race-free equation)
# ---------------------------------------------------------------------------

def calc_egfr(
    creatinine: float,
    age: int,
    sex: str,
) -> CalculatorResult:
    """Calculate eGFR using the CKD-EPI 2021 equation (race-free).

    Args:
        creatinine: Serum creatinine in mg/dL.
        age: Patient age in years.
        sex: "male" or "female".

    Returns:
        CalculatorResult with eGFR in mL/min/1.73m².
    """
    if sex.lower() == "female":
        kappa = 0.7
        alpha = -0.241
        sex_coeff = 1.012
    else:
        kappa = 0.9
        alpha = -0.302
        sex_coeff = 1.0

    cr_ratio = creatinine / kappa
    egfr = 142 * (min(cr_ratio, 1.0) ** alpha) * (max(cr_ratio, 1.0) ** -1.200) * (0.9938 ** age) * sex_coeff

    if egfr >= 90:
        interp = "Normal or high (G1)"
    elif egfr >= 60:
        interp = "Mildly decreased (G2)"
    elif egfr >= 45:
        interp = "Mildly to moderately decreased (G3a)"
    elif egfr >= 30:
        interp = "Moderately to severely decreased (G3b)"
    elif egfr >= 15:
        interp = "Severely decreased (G4)"
    else:
        interp = "Kidney failure (G5)"

    return CalculatorResult(
        calculator="eGFR (CKD-EPI 2021)",
        score=round(egfr, 1),
        interpretation=interp,
        details={"creatinine_mg_dl": creatinine, "age": age, "sex": sex, "unit": "mL/min/1.73m²"},
    )


# ---------------------------------------------------------------------------
# CHA₂DS₂-VASc — stroke risk in atrial fibrillation
# ---------------------------------------------------------------------------

def calc_cha2ds2_vasc(
    age: int,
    sex: str,
    has_chf: bool = False,
    has_hypertension: bool = False,
    has_stroke_tia: bool = False,
    has_vascular_disease: bool = False,
    has_diabetes: bool = False,
) -> CalculatorResult:
    """Calculate CHA₂DS₂-VASc score for stroke risk stratification.

    Args:
        age: Patient age.
        sex: "male" or "female".
        has_chf: Congestive heart failure history.
        has_hypertension: Hypertension.
        has_stroke_tia: Prior stroke, TIA, or thromboembolism.
        has_vascular_disease: Prior MI, PAD, or aortic plaque.
        has_diabetes: Diabetes mellitus.

    Returns:
        CalculatorResult with score 0-9.
    """
    score = 0
    if has_chf:
        score += 1
    if has_hypertension:
        score += 1
    if has_stroke_tia:
        score += 2
    if has_vascular_disease:
        score += 1
    if has_diabetes:
        score += 1
    if sex.lower() == "female":
        score += 1
    if age >= 75:
        score += 2
    elif age >= 65:
        score += 1

    if score == 0:
        interp = "Low risk — no anticoagulation recommended"
    elif score == 1:
        interp = "Low-moderate risk — consider anticoagulation"
    else:
        interp = f"Moderate-high risk (score {score}) — oral anticoagulation recommended"

    return CalculatorResult(
        calculator="CHA₂DS₂-VASc",
        score=score,
        interpretation=interp,
        details={
            "age": age, "sex": sex,
            "chf": has_chf, "hypertension": has_hypertension,
            "stroke_tia": has_stroke_tia, "vascular_disease": has_vascular_disease,
            "diabetes": has_diabetes,
        },
    )


# ---------------------------------------------------------------------------
# MELD — Model for End-Stage Liver Disease
# ---------------------------------------------------------------------------

def calc_meld(
    bilirubin: float,
    inr: float,
    creatinine: float,
    sodium: float = 137.0,
    on_dialysis: bool = False,
) -> CalculatorResult:
    """Calculate MELD-Na score for liver disease severity.

    Args:
        bilirubin: Total bilirubin in mg/dL (min 1.0 for formula).
        inr: International Normalized Ratio (min 1.0 for formula).
        creatinine: Serum creatinine in mg/dL (min 1.0, max 4.0 for formula).
        sodium: Serum sodium in mEq/L (clamped 125-137 for formula).
        on_dialysis: If True, creatinine is set to 4.0.

    Returns:
        CalculatorResult with MELD-Na score.
    """
    bili = max(bilirubin, 1.0)
    inr_val = max(inr, 1.0)
    cr = 4.0 if on_dialysis else max(min(creatinine, 4.0), 1.0)
    na = max(min(sodium, 137.0), 125.0)

    meld = 10 * (
        0.957 * math.log(cr)
        + 0.378 * math.log(bili)
        + 1.120 * math.log(inr_val)
        + 0.643
    )
    meld = round(meld)
    meld = max(meld, 6)
    meld = min(meld, 40)

    # MELD-Na adjustment
    meld_na = meld + 1.32 * (137 - na) - 0.033 * meld * (137 - na)
    meld_na = round(max(min(meld_na, 40), 6))

    if meld_na < 10:
        interp = "Low severity — 90-day mortality ~2%"
    elif meld_na < 20:
        interp = "Moderate severity — 90-day mortality ~6%"
    elif meld_na < 30:
        interp = "High severity — 90-day mortality ~20%"
    else:
        interp = "Very high severity — 90-day mortality >50%"

    return CalculatorResult(
        calculator="MELD-Na",
        score=meld_na,
        interpretation=interp,
        details={
            "bilirubin": bilirubin, "inr": inr, "creatinine": creatinine,
            "sodium": sodium, "on_dialysis": on_dialysis,
            "meld_base": meld, "meld_na": meld_na,
        },
    )


# ---------------------------------------------------------------------------
# Wells Score — DVT probability
# ---------------------------------------------------------------------------

def calc_wells_dvt(
    active_cancer: bool = False,
    paralysis_or_cast: bool = False,
    bedridden_3_days: bool = False,
    tenderness_along_veins: bool = False,
    entire_leg_swollen: bool = False,
    calf_swelling_gt_3cm: bool = False,
    pitting_edema: bool = False,
    collateral_veins: bool = False,
    previous_dvt: bool = False,
    alternative_diagnosis_likely: bool = False,
) -> CalculatorResult:
    """Calculate Wells Score for DVT probability.

    Returns:
        CalculatorResult with score and risk category.
    """
    score = 0
    if active_cancer:
        score += 1
    if paralysis_or_cast:
        score += 1
    if bedridden_3_days:
        score += 1
    if tenderness_along_veins:
        score += 1
    if entire_leg_swollen:
        score += 1
    if calf_swelling_gt_3cm:
        score += 1
    if pitting_edema:
        score += 1
    if collateral_veins:
        score += 1
    if previous_dvt:
        score += 1
    if alternative_diagnosis_likely:
        score -= 2

    if score <= 0:
        interp = "Low probability (~5%) — consider D-dimer"
    elif score <= 2:
        interp = "Moderate probability (~17%) — consider ultrasound or D-dimer"
    else:
        interp = "High probability (~53%) — ultrasound recommended"

    return CalculatorResult(
        calculator="Wells Score (DVT)",
        score=score,
        interpretation=interp,
        details={
            "active_cancer": active_cancer, "paralysis_or_cast": paralysis_or_cast,
            "bedridden_3_days": bedridden_3_days, "tenderness_along_veins": tenderness_along_veins,
            "entire_leg_swollen": entire_leg_swollen, "calf_swelling_gt_3cm": calf_swelling_gt_3cm,
            "pitting_edema": pitting_edema, "collateral_veins": collateral_veins,
            "previous_dvt": previous_dvt, "alternative_diagnosis_likely": alternative_diagnosis_likely,
        },
    )


# ---------------------------------------------------------------------------
# CURB-65 — pneumonia severity
# ---------------------------------------------------------------------------

def calc_curb65(
    confusion: bool = False,
    bun_gt_19: bool = False,
    respiratory_rate_ge_30: bool = False,
    systolic_bp_lt_90: bool = False,
    diastolic_bp_le_60: bool = False,
    age_ge_65: bool = False,
) -> CalculatorResult:
    """Calculate CURB-65 score for community-acquired pneumonia severity.

    Args:
        confusion: New mental confusion.
        bun_gt_19: Blood urea nitrogen > 19 mg/dL (>7 mmol/L).
        respiratory_rate_ge_30: Respiratory rate ≥ 30/min.
        systolic_bp_lt_90: Systolic BP < 90 mmHg.
        diastolic_bp_le_60: Diastolic BP ≤ 60 mmHg.
        age_ge_65: Age ≥ 65 years.

    Returns:
        CalculatorResult with score 0-5.
    """
    score = 0
    if confusion:
        score += 1
    if bun_gt_19:
        score += 1
    if respiratory_rate_ge_30:
        score += 1
    if systolic_bp_lt_90 or diastolic_bp_le_60:
        score += 1
    if age_ge_65:
        score += 1

    if score <= 1:
        interp = "Low severity — consider outpatient treatment (mortality ~1.5%)"
    elif score == 2:
        interp = "Moderate severity — consider short inpatient or supervised outpatient (mortality ~9.2%)"
    elif score == 3:
        interp = "Severe — hospitalize, consider ICU (mortality ~22%)"
    else:
        interp = "Very severe — ICU admission recommended (mortality ~30%+)"

    return CalculatorResult(
        calculator="CURB-65",
        score=score,
        interpretation=interp,
        details={
            "confusion": confusion, "bun_gt_19": bun_gt_19,
            "respiratory_rate_ge_30": respiratory_rate_ge_30,
            "low_bp": systolic_bp_lt_90 or diastolic_bp_le_60,
            "age_ge_65": age_ge_65,
        },
    )
