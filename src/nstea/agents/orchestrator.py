"""Phase 5 orchestrator — hardened pipeline with temporal analysis + symbolic constraints.

Pipeline stages:
1. RAG retrieval + Temporal analysis (parallel)
2. Pre-compute safety context (rule engine + constraint engine on patient's current meds)
3. LLM reasoning (with guideline + safety + temporal context) — with retry + timeout
4. Post-hoc safety check (rule engine + symbolic constraint engine)
5. Confidence gating (human-in-the-loop decision)
"""

from __future__ import annotations

import asyncio
import time
import uuid

import structlog

from nstea.agents.confidence_agent import evaluate_confidence
from nstea.agents.safety_agent import run_safety_check
from nstea.agents.temporal_agent import TemporalResult, run_temporal_analysis
from nstea.core.logging import bind_correlation_id
from nstea.models.analysis import (
    AnalysisResponse,
    ConfidenceScore,
    DiagnosisOutput,
    ReasoningStep,
    SafetyFlag,
)
from nstea.models.patient import PatientInput
from nstea.retrieval.context_builder import build_context
from nstea.retrieval.vector_store import SearchResult
from nstea.safety import RuleEngine
from nstea.safety.constraint_engine import SymbolicConstraintEngine

logger = structlog.stdlib.get_logger(__name__)

# --- Configuration ---
LLM_TIMEOUT_S = 120  # Max seconds for a single LLM call
LLM_MAX_RETRIES = 2  # Number of retries on LLM failure
LLM_RETRY_BACKOFF = 2.0  # Exponential backoff base (seconds)
PIPELINE_TIMEOUT_S = 180  # Max total pipeline time


def _get_vector_store():
    """Lazily load the vector store (same singleton as guideline_search tool)."""
    from nstea.tools.guideline_search import _get_store

    return _get_store()


def retrieve_guidelines(patient: PatientInput) -> list[SearchResult]:
    """Retrieve relevant guidelines based on patient conditions and query."""
    store = _get_vector_store()

    query_parts = []
    if patient.clinician_query:
        query_parts.append(patient.clinician_query)
    for c in patient.conditions:
        query_parts.append(c.name)
    for s in patient.symptoms:
        query_parts.append(s.description)
    for a in patient.allergies:
        query_parts.append(f"{a.substance} allergy")

    query = " ".join(query_parts) if query_parts else "general clinical assessment"
    return store.search(query, top_k=5, score_threshold=0.25)


def pre_compute_safety_context(patient: PatientInput) -> str:
    """Run the rule engine + constraint engine against patient's existing data.

    This generates a safety summary that gets injected into the LLM prompt,
    so the model is aware of relevant contraindications BEFORE generating recommendations.
    """
    engine = RuleEngine()
    constraint_engine = SymbolicConstraintEngine()

    lines = ["=== SAFETY PRE-CHECK RESULTS ==="]

    # --- YAML rule engine check on current medications ---
    current_meds = [m.name for m in patient.medications]
    if current_meds:
        result = engine.check_proposed_drugs(patient, current_meds)
        if result.is_safe and not result.violations:
            lines.append("Rule engine: No safety issues with current medications.")
        else:
            for v in result.violations:
                icon = "CRITICAL" if v.severity == "critical" else "WARNING"
                lines.append(f"[{icon}] {v.message}")
                if v.blocked_drug:
                    lines.append(f"  → Drug: {v.blocked_drug}")
    else:
        lines.append("No current medications to check.")

    # --- Symbolic constraint engine check on current medications ---
    if current_meds:
        constraint_result = constraint_engine.validate(current_meds, patient)
        if constraint_result.violations:
            lines.append("\nKNOWLEDGE GRAPH CONSTRAINTS:")
            for cv in constraint_result.violations:
                icon = "CRITICAL" if cv.severity == "critical" else "WARNING"
                lines.append(f"[{icon}] {cv.description}")
                if cv.alternative:
                    lines.append(f"  → Alternative: {cv.alternative}")
        if constraint_result.unmapped_actions:
            lines.append(f"\nUNMAPPED DRUGS (not in knowledge graph): {', '.join(constraint_result.unmapped_actions)}")

    # Always include allergy summary for LLM awareness
    if patient.allergies:
        lines.append("\nPATIENT ALLERGIES (DO NOT prescribe these or cross-reactive drugs):")
        for a in patient.allergies:
            lines.append(f"  - {a.substance} ({a.severity}): {a.reaction or 'unknown reaction'}")

    return "\n".join(lines)


async def run_pipeline_async(patient: PatientInput) -> AnalysisResponse:
    """Run the full analysis pipeline with retry, timeout, and error boundaries.

    Args:
        patient: Structured patient input data.

    Returns:
        Final AnalysisResponse after all pipeline stages.
    """
    correlation_id = str(uuid.uuid4())[:8]
    start = time.time()
    stage_timings: dict[str, float] = {}

    bind_correlation_id(correlation_id)
    logger.info("pipeline_start", patient_id=patient.patient_id)

    # --- Stage 1: RAG retrieval + Temporal analysis (parallel with error boundaries) ---
    t0 = time.time()

    # Run RAG and Temporal in parallel
    async def _rag_stage():
        try:
            sr = retrieve_guidelines(patient)
            ps = patient.to_clinical_summary()
            gc = build_context(ps, sr)
            return sr, gc
        except Exception:
            logger.exception("rag_retrieval_failed")
            return [], "RAG retrieval failed — proceeding without guideline context."

    async def _temporal_stage():
        try:
            return run_temporal_analysis(patient)
        except Exception:
            logger.exception("temporal_analysis_failed")
            return TemporalResult(insights=["Temporal analysis unavailable."])

    rag_task = asyncio.create_task(_rag_stage())
    temporal_task = asyncio.create_task(_temporal_stage())

    (search_results, guideline_context), temporal_result = await asyncio.gather(
        rag_task, temporal_task
    )
    stage_timings["rag_and_temporal"] = time.time() - t0

    # --- Stage 2: Safety pre-check (with error boundary) ---
    t0 = time.time()
    try:
        safety_context = pre_compute_safety_context(patient)
    except Exception:
        logger.exception("safety_precheck_failed")
        safety_context = "Safety pre-check failed — exercise extra caution."
    stage_timings["safety_precheck"] = time.time() - t0

    # --- Stage 3: LLM reasoning (with retry + timeout) ---
    from nstea.agents.reasoning_agent_v1 import analyze_patient_v1_async

    # Combine guideline + temporal context for the LLM
    temporal_context = temporal_result.to_context_string()
    combined_context = f"{guideline_context}\n\n{temporal_context}"

    analysis = None
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 2):  # +2 because range is exclusive and attempt 1 is first try
        t0 = time.time()
        try:
            analysis = await asyncio.wait_for(
                analyze_patient_v1_async(patient, combined_context, safety_context),
                timeout=LLM_TIMEOUT_S,
            )
            stage_timings[f"llm_attempt_{attempt}"] = time.time() - t0
            logger.info("llm_success", attempt=attempt, elapsed=round(time.time() - t0, 1))
            break
        except asyncio.TimeoutError:
            stage_timings[f"llm_attempt_{attempt}"] = time.time() - t0
            last_error = f"LLM timeout after {LLM_TIMEOUT_S}s (attempt {attempt})"
            logger.warning("llm_timeout", attempt=attempt)
        except Exception as e:
            stage_timings[f"llm_attempt_{attempt}"] = time.time() - t0
            last_error = f"LLM error: {type(e).__name__}: {e} (attempt {attempt})"
            logger.warning("llm_error", attempt=attempt, error=str(e))

        if attempt <= LLM_MAX_RETRIES:
            backoff = LLM_RETRY_BACKOFF ** attempt
            logger.info("llm_retry", backoff=round(backoff, 1))
            await asyncio.sleep(backoff)

    # If all LLM attempts failed, produce a degraded response
    if analysis is None:
        logger.error("llm_all_attempts_failed", last_error=last_error)
        analysis = AnalysisResponse(
            analysis_id=correlation_id,
            patient_id=patient.patient_id,
            diagnosis=DiagnosisOutput(primary="LLM UNAVAILABLE — analysis could not be completed"),
            confidence=ConfidenceScore(overall=0.0, evidence_strength=0.0, model_certainty=0.0),
            requires_human_review=True,
            escalation_reason=f"All LLM attempts failed: {last_error}",
        )

    # --- Stage 4: Post-hoc safety check (with error boundary) ---
    t0 = time.time()
    try:
        analysis = run_safety_check(patient, analysis)
    except Exception:
        logger.exception("safety_postcheck_failed")
        analysis.requires_human_review = True
        analysis.escalation_reason = (analysis.escalation_reason or "") + " | Safety post-check failed"

    # --- Stage 4b: Symbolic constraint engine post-check ---
    try:
        constraint_engine = SymbolicConstraintEngine()
        proposed_drugs = [
            rec.action for rec in analysis.recommendations if rec.category == "medication"
        ]
        if proposed_drugs:
            constraint_result = constraint_engine.validate(proposed_drugs, patient)

            # Convert constraint violations to SafetyFlags
            critical_drugs: set[str] = set()
            existing_messages = {f.message for f in analysis.safety_flags}
            for cv in constraint_result.violations:
                msg = f"[CONSTRAINT ENGINE] {cv.description}"
                if msg not in existing_messages:
                    analysis.safety_flags.append(SafetyFlag(
                        level=cv.severity,
                        message=msg,
                        related_recommendation=cv.action_blocked,
                    ))
                if cv.severity == "critical":
                    critical_drugs.add(cv.action_blocked.lower())

            # Remove recommendations blocked by constraint engine
            if critical_drugs:
                safe_recs = []
                for rec in analysis.recommendations:
                    action_lower = rec.action.lower()
                    if any(drug in action_lower for drug in critical_drugs):
                        analysis.safety_flags.append(SafetyFlag(
                            level="critical",
                            message=f"[CONSTRAINT ENGINE] REMOVED recommendation: '{rec.action}'",
                            related_recommendation=rec.action,
                        ))
                    else:
                        safe_recs.append(rec)
                analysis.recommendations = safe_recs

            # Flag unmapped actions for mandatory human review
            if constraint_result.unmapped_actions:
                analysis.safety_flags.append(SafetyFlag(
                    level="warning",
                    message=f"[CONSTRAINT ENGINE] Unmapped drugs (not in KG): {', '.join(constraint_result.unmapped_actions)}. Human review mandatory.",
                ))

            if constraint_result.has_critical:
                analysis.requires_human_review = True
                existing = analysis.escalation_reason or ""
                analysis.escalation_reason = f"{existing} | CONSTRAINT ENGINE: {constraint_result.review_reason}".lstrip(" |")

            # Add guideline alignment info
            if constraint_result.guideline_alignment:
                for ga in constraint_result.guideline_alignment:
                    analysis.safety_flags.append(SafetyFlag(
                        level="info",
                        message=f"[GUIDELINE] {ga.get('guideline', '')}: recommends {ga.get('drug', '')} ({ga.get('strength', '')})",
                    ))
    except Exception:
        logger.exception("constraint_engine_postcheck_failed")

    stage_timings["safety_postcheck"] = time.time() - t0

    # --- Stage 5: Confidence gating (with error boundary) ---
    t0 = time.time()
    try:
        analysis = evaluate_confidence(analysis)
    except Exception:
        logger.exception("confidence_gate_failed")
        analysis.requires_human_review = True
    stage_timings["confidence"] = time.time() - t0

    elapsed = time.time() - start

    # Inject pipeline metadata into reasoning steps
    timing_summary = " | ".join(f"{k}={v:.1f}s" for k, v in stage_timings.items())
    temporal_info = f"temporal_insights={len(temporal_result.insights)}, from_cache={temporal_result.from_cache}"
    analysis.reasoning_steps.insert(
        0,
        ReasoningStep(
            step_number=0,
            description="Pipeline metadata",
            input_summary=f"Retrieved {len(search_results)} guideline chunks | {temporal_info} | correlation_id={correlation_id}",
            output_summary=f"Pipeline completed in {elapsed:.1f}s | {timing_summary}",
        ),
    )

    logger.info(
        "pipeline_complete",
        elapsed=round(elapsed, 1),
        stage_timings=stage_timings,
        requires_human_review=analysis.requires_human_review,
    )

    return analysis


def run_pipeline(patient: PatientInput) -> AnalysisResponse:
    """Synchronous wrapper for run_pipeline_async."""
    return asyncio.run(run_pipeline_async(patient))
