"""Phase 0 clinical reasoning agent using Agno framework.

Simple LLM-only analysis — no RAG, no tool use.
Used for baseline comparison and quick analysis.
"""

import json
import re
import uuid

from agno.agent import Agent
from agno.models.ollama import Ollama

from nstea.agents.prompts.reasoning import REASONING_PROMPT_V01, SYSTEM_PROMPT
from nstea.config import settings
from nstea.models.analysis import AnalysisResponse, ConfidenceScore, DiagnosisOutput
from nstea.models.patient import PatientInput


def _build_model():
    """Build the Agno model based on config."""
    if settings.model_provider == "ollama":
        kwargs = {
            "id": settings.model_id,
            "options": {"temperature": settings.temperature},
        }
        if settings.ollama_api_key:
            # Cloud: host auto-set to https://ollama.com by Agno
            kwargs["api_key"] = settings.ollama_api_key
        else:
            kwargs["host"] = settings.ollama_host
        return Ollama(**kwargs)
    elif settings.model_provider == "huggingface":
        from agno.models.huggingface import HuggingFace

        return HuggingFace(
            id=settings.model_id,
            api_key=settings.hf_token or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    else:
        raise ValueError(f"Unsupported model_provider: {settings.model_provider}")


def _get_agent() -> Agent:
    """Create a Phase 0 reasoning agent."""
    return Agent(
        name="ClinicalReasonerV0",
        model=_build_model(),
        instructions=SYSTEM_PROMPT,
        description="Analyzes patient clinical data and provides structured diagnostic reasoning.",
        markdown=False,
    )


async def analyze_patient_async(patient: PatientInput) -> AnalysisResponse:
    """Run Phase 0 clinical analysis (LLM only).

    Args:
        patient: Structured patient input data.

    Returns:
        Parsed AnalysisResponse with diagnosis, recommendations, safety flags.
    """
    analysis_id = str(uuid.uuid4())[:8]
    clinical_summary = patient.to_clinical_summary()
    prompt_text = REASONING_PROMPT_V01.format(patient_summary=clinical_summary)

    agent = _get_agent()
    response = await agent.arun(prompt_text)
    raw_text = response.content or ""

    return _parse_response(raw_text, analysis_id, patient.patient_id)


def analyze_patient(patient: PatientInput) -> AnalysisResponse:
    """Synchronous wrapper for analyze_patient_async."""
    import asyncio

    return asyncio.run(analyze_patient_async(patient))


def _parse_response(raw_text: str, analysis_id: str, patient_id: str) -> AnalysisResponse:
    """Parse LLM JSON output into AnalysisResponse, with fallback for malformed output."""
    text = raw_text.strip()

    # Strip <think>...</think> blocks (qwen3/deepseek thinking output)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try to extract JSON object from mixed text
    json_text = text
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            json_text = match.group(0)
        else:
            json_text = None

    data = None
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            pass

    if data is None:
        return AnalysisResponse(
            analysis_id=analysis_id,
            patient_id=patient_id,
            diagnosis=DiagnosisOutput(primary="FREE-TEXT RESPONSE (no JSON)"),
            confidence=ConfidenceScore(overall=0.0, evidence_strength=0.0, model_certainty=0.0),
            requires_human_review=True,
            escalation_reason="LLM output could not be parsed as JSON",
            raw_llm_output=raw_text,
        )

    try:
        return AnalysisResponse(
            analysis_id=analysis_id,
            patient_id=patient_id,
            diagnosis=data.get("diagnosis", {"primary": "Unknown"}),
            recommendations=data.get("recommendations", []),
            reasoning_steps=data.get("reasoning_steps", []),
            safety_flags=data.get("safety_flags", []),
            confidence=data.get("confidence", {"overall": 0.5, "evidence_strength": 0.5, "model_certainty": 0.5}),
            requires_human_review=data.get("requires_human_review", True),
            escalation_reason=data.get("escalation_reason"),
            raw_llm_output=raw_text,
        )
    except Exception:
        return AnalysisResponse(
            analysis_id=analysis_id,
            patient_id=patient_id,
            diagnosis=DiagnosisOutput(primary="VALIDATION ERROR: LLM output did not match schema"),
            confidence=ConfidenceScore(overall=0.0, evidence_strength=0.0, model_certainty=0.0),
            requires_human_review=True,
            escalation_reason="LLM output passed JSON parse but failed schema validation",
            raw_llm_output=raw_text,
        )
