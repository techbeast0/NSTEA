"""Phase 1 clinical reasoning agent using Agno framework.

Upgrades Phase 0 with pre-computed context injection:
- RAG guideline context injected into prompt (pre-retrieved by orchestrator)
- Safety check results injected into prompt (pre-computed by orchestrator)
- No tool calling — context is pre-built for reliability with local models
"""

import json
import re
import uuid

from agno.agent import Agent
from agno.models.ollama import Ollama

from nstea.agents.prompts.reasoning_v1 import REASONING_PROMPT_V1, SYSTEM_PROMPT_V1
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
    """Create a Phase 1 reasoning agent with pre-computed context."""
    return Agent(
        name="ClinicalReasonerV1",
        model=_build_model(),
        instructions=SYSTEM_PROMPT_V1,
        description=(
            "Phase 1 clinical reasoning agent with RAG-grounded guidelines "
            "and hard safety rule checking via pre-computed context."
        ),
        markdown=False,
    )


async def analyze_patient_v1_async(
    patient: PatientInput,
    guideline_context: str = "",
    safety_context: str = "",
) -> AnalysisResponse:
    """Run Phase 1 clinical analysis with pre-computed RAG + safety context.

    Args:
        patient: Structured patient input.
        guideline_context: Pre-retrieved guideline context (from RAG).
        safety_context: Pre-computed safety check results (from rule engine).

    Returns:
        Parsed AnalysisResponse.
    """
    analysis_id = str(uuid.uuid4())[:8]
    clinical_summary = patient.to_clinical_summary()
    patient_json = patient.model_dump_json()

    if not guideline_context:
        guideline_context = "No guidelines retrieved for this case."
    if not safety_context:
        safety_context = "No pre-computed safety checks available."

    prompt_text = REASONING_PROMPT_V1.format(
        patient_summary=clinical_summary,
        guideline_context=guideline_context,
        safety_context=safety_context,
        patient_json=patient_json,
    )

    agent = _get_agent()
    response = await agent.arun(prompt_text)
    raw_text = response.content or ""

    return _parse_response(raw_text, analysis_id, patient.patient_id)


def analyze_patient_v1(
    patient: PatientInput,
    guideline_context: str = "",
    safety_context: str = "",
) -> AnalysisResponse:
    """Synchronous wrapper."""
    import asyncio

    return asyncio.run(
        analyze_patient_v1_async(patient, guideline_context, safety_context)
    )


def _parse_response(
    raw_text: str, analysis_id: str, patient_id: str
) -> AnalysisResponse:
    """Parse LLM JSON output into AnalysisResponse with fallback."""
    text = raw_text.strip()

    # Strip <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Extract JSON
    json_text = text
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        json_text = match.group(0) if match else None

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
            confidence=ConfidenceScore(
                overall=0.0, evidence_strength=0.0, model_certainty=0.0
            ),
            requires_human_review=True,
            escalation_reason="LLM output could not be parsed as JSON",
            raw_llm_output=raw_text,
        )

    try:
        # Normalize common key variations the LLM might produce
        diagnosis_raw = data.get("diagnosis", data.get("primary_diagnosis", {"primary": "Unknown"}))
        if isinstance(diagnosis_raw, str):
            diagnosis_raw = {"primary": diagnosis_raw}

        recs_raw = data.get("recommendations", [])
        # Normalize recommendation fields
        normalized_recs = []
        for r in recs_raw:
            if isinstance(r, str):
                normalized_recs.append({"action": r, "category": "general", "urgency": "routine", "rationale": ""})
            elif isinstance(r, dict):
                r.setdefault("action", r.get("description", r.get("recommendation", "Unknown")))
                r.setdefault("category", "general")
                r.setdefault("urgency", "routine")
                r.setdefault("rationale", "")
                normalized_recs.append(r)

        steps_raw = data.get("reasoning_steps", data.get("reasoning", []))
        normalized_steps = []
        for i, s in enumerate(steps_raw):
            if isinstance(s, str):
                normalized_steps.append({"step_number": i + 1, "description": s, "input_summary": "", "output_summary": ""})
            elif isinstance(s, dict):
                s.setdefault("step_number", i + 1)
                s.setdefault("description", s.get("step", s.get("name", f"Step {i+1}")))
                s.setdefault("input_summary", s.get("input", ""))
                s.setdefault("output_summary", s.get("output", s.get("conclusion", "")))
                normalized_steps.append(s)

        conf_raw = data.get(
            "confidence",
            {"overall": 0.5, "evidence_strength": 0.5, "model_certainty": 0.5},
        )
        if isinstance(conf_raw, (int, float)):
            conf_raw = {"overall": conf_raw, "evidence_strength": conf_raw, "model_certainty": conf_raw}

        return AnalysisResponse(
            analysis_id=analysis_id,
            patient_id=patient_id,
            diagnosis=diagnosis_raw,
            recommendations=normalized_recs,
            reasoning_steps=normalized_steps,
            safety_flags=data.get("safety_flags", []),
            confidence=conf_raw,
            requires_human_review=data.get("requires_human_review", True),
            escalation_reason=data.get("escalation_reason"),
            raw_llm_output=raw_text,
        )
    except Exception:
        return AnalysisResponse(
            analysis_id=analysis_id,
            patient_id=patient_id,
            diagnosis=DiagnosisOutput(
                primary="VALIDATION ERROR: LLM output did not match schema"
            ),
            confidence=ConfidenceScore(
                overall=0.0, evidence_strength=0.0, model_certainty=0.0
            ),
            requires_human_review=True,
            escalation_reason="LLM output passed JSON parse but failed schema validation",
            raw_llm_output=raw_text,
        )
