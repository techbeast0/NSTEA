"""Feedback API routes — clinician review of analysis results."""

from __future__ import annotations

from fastapi import APIRouter

from nstea.models.feedback import FeedbackInput, FeedbackRecord
from nstea.services.feedback_service import (
    get_feedback_for_analysis,
    get_feedback_summary,
    submit_feedback,
)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackRecord)
async def post_feedback(feedback: FeedbackInput) -> FeedbackRecord:
    """Submit clinician feedback on an analysis."""
    return submit_feedback(feedback)


@router.get("/feedback/{analysis_id}", response_model=list[FeedbackRecord])
async def get_feedback(analysis_id: str) -> list[FeedbackRecord]:
    """Get all feedback for a specific analysis."""
    return get_feedback_for_analysis(analysis_id)


@router.get("/feedback-summary")
async def feedback_summary():
    """Get aggregate feedback statistics."""
    return get_feedback_summary()
