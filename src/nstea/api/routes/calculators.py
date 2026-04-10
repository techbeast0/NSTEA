"""Calculator API routes — clinical scoring tools."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nstea.tools.lab_calculator import (
    calc_cha2ds2_vasc,
    calc_curb65,
    calc_egfr,
    calc_meld,
    calc_wells_dvt,
)

router = APIRouter()


class EGFRRequest(BaseModel):
    creatinine: float
    age: int
    sex: str


class CHA2DS2VAScRequest(BaseModel):
    age: int
    sex: str
    has_chf: bool = False
    has_hypertension: bool = False
    has_stroke_tia: bool = False
    has_vascular_disease: bool = False
    has_diabetes: bool = False


class MELDRequest(BaseModel):
    bilirubin: float
    inr: float
    creatinine: float
    sodium: float = 137.0
    on_dialysis: bool = False


class WellsRequest(BaseModel):
    active_cancer: bool = False
    paralysis_or_cast: bool = False
    bedridden_3_days: bool = False
    tenderness_along_veins: bool = False
    entire_leg_swollen: bool = False
    calf_swelling_gt_3cm: bool = False
    pitting_edema: bool = False
    collateral_veins: bool = False
    previous_dvt: bool = False
    alternative_diagnosis_likely: bool = False


class CURB65Request(BaseModel):
    confusion: bool = False
    bun_gt_19: bool = False
    respiratory_rate_ge_30: bool = False
    systolic_bp_lt_90: bool = False
    diastolic_bp_le_60: bool = False
    age_ge_65: bool = False


@router.post("/calculators/egfr")
async def calculate_egfr(req: EGFRRequest):
    try:
        result = calc_egfr(req.creatinine, req.age, req.sex)
        return {"calculator": result.calculator, "score": result.score,
                "interpretation": result.interpretation, "details": result.details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calculators/cha2ds2vasc")
async def calculate_cha2ds2vasc(req: CHA2DS2VAScRequest):
    try:
        result = calc_cha2ds2_vasc(
            req.age, req.sex, req.has_chf, req.has_hypertension,
            req.has_stroke_tia, req.has_vascular_disease, req.has_diabetes,
        )
        return {"calculator": result.calculator, "score": result.score,
                "interpretation": result.interpretation, "details": result.details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calculators/meld")
async def calculate_meld(req: MELDRequest):
    try:
        result = calc_meld(req.bilirubin, req.inr, req.creatinine, req.sodium, req.on_dialysis)
        return {"calculator": result.calculator, "score": result.score,
                "interpretation": result.interpretation, "details": result.details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calculators/wells")
async def calculate_wells(req: WellsRequest):
    try:
        result = calc_wells_dvt(
            req.active_cancer, req.paralysis_or_cast, req.bedridden_3_days,
            req.tenderness_along_veins, req.entire_leg_swollen, req.calf_swelling_gt_3cm,
            req.pitting_edema, req.collateral_veins, req.previous_dvt,
            req.alternative_diagnosis_likely,
        )
        return {"calculator": result.calculator, "score": result.score,
                "interpretation": result.interpretation, "details": result.details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calculators/curb65")
async def calculate_curb65(req: CURB65Request):
    try:
        result = calc_curb65(
            req.confusion, req.bun_gt_19, req.respiratory_rate_ge_30,
            req.systolic_bp_lt_90, req.diastolic_bp_le_60, req.age_ge_65,
        )
        return {"calculator": result.calculator, "score": result.score,
                "interpretation": result.interpretation, "details": result.details}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
