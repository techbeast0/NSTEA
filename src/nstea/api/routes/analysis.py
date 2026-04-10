"""Analysis API routes — Phase 1 clinical reasoning pipeline."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from nstea.agents.orchestrator import run_pipeline_async
from nstea.models.analysis import AnalysisResponse
from nstea.models.patient import PatientInput

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient(patient: PatientInput) -> AnalysisResponse:
    """Run the full Phase 1 clinical analysis pipeline.

    Accepts structured patient data and returns a comprehensive
    clinical analysis with diagnosis, recommendations, safety
    flags, and confidence scoring.
    """
    try:
        result = await run_pipeline_async(patient)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {type(e).__name__}: {e}",
        )


@router.post("/analyze/quick", response_model=AnalysisResponse)
async def analyze_patient_quick(patient: PatientInput) -> AnalysisResponse:
    """Run Phase 0 analysis (LLM only, no RAG or safety rules).

    Faster but less safe — useful for testing and comparison.
    """
    from nstea.agents.reasoning_agent import analyze_patient_async

    try:
        result = await analyze_patient_async(patient)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {type(e).__name__}: {e}",
        )
