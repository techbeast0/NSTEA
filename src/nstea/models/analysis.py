"""Analysis response models for NS-TEA."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DifferentialDx(BaseModel):
    diagnosis: str
    probability: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class DiagnosisOutput(BaseModel):
    primary: str
    differential: list[DifferentialDx] = Field(default_factory=list)


class Recommendation(BaseModel):
    action: str
    category: Literal["medication", "test", "procedure", "referral", "monitoring"]
    urgency: Literal["stat", "urgent", "routine"]
    rationale: str
    guideline_source: Optional[str] = None


class ReasoningStep(BaseModel):
    step_number: int
    description: str
    input_summary: str
    output_summary: str


class SafetyFlag(BaseModel):
    level: Literal["info", "warning", "critical"]
    message: str
    related_recommendation: Optional[str] = None


class ConfidenceScore(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    model_certainty: float = Field(ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Complete clinical analysis output."""

    analysis_id: str
    patient_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    diagnosis: DiagnosisOutput
    recommendations: list[Recommendation] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    confidence: ConfidenceScore
    requires_human_review: bool = True
    escalation_reason: Optional[str] = None

    raw_llm_output: Optional[str] = None
