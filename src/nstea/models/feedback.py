"""Feedback models for clinician review of analysis results."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeedbackInput(BaseModel):
    """Clinician feedback on an analysis."""

    analysis_id: str
    clinician_id: str
    verdict: Literal["accept", "modify", "reject"]
    notes: str = ""
    modified_diagnosis: Optional[str] = None
    modified_recommendations: list[str] = Field(default_factory=list)


class FeedbackRecord(FeedbackInput):
    """Stored feedback record with metadata."""

    feedback_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
