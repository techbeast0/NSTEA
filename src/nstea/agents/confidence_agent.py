"""Confidence agent — scores and gates analysis for human-in-the-loop review.

Evaluates the LLM analysis quality and determines whether automated
output is acceptable or requires human clinician review.
"""

from __future__ import annotations

from nstea.models.analysis import AnalysisResponse


# Thresholds
CONFIDENCE_THRESHOLD = 0.65
MIN_REASONING_STEPS = 2
MIN_RECOMMENDATIONS = 1


def evaluate_confidence(analysis: AnalysisResponse) -> AnalysisResponse:
    """Evaluate the quality and safety of an analysis and gate for human review.

    Checks:
    1. Overall confidence above threshold
    2. Minimum reasoning steps present
    3. At least one recommendation
    4. No critical safety flags from rule engine
    5. Diagnosis is not a fallback/error

    Args:
        analysis: The analysis to evaluate.

    Returns:
        Amended AnalysisResponse with updated requires_human_review and
        escalation_reason based on confidence evaluation.
    """
    escalation_reasons: list[str] = []

    # 1. Low confidence
    if analysis.confidence.overall < CONFIDENCE_THRESHOLD:
        escalation_reasons.append(
            f"Low confidence ({analysis.confidence.overall:.2f} < {CONFIDENCE_THRESHOLD})"
        )

    # 2. Insufficient reasoning
    if len(analysis.reasoning_steps) < MIN_REASONING_STEPS:
        escalation_reasons.append(
            f"Insufficient reasoning ({len(analysis.reasoning_steps)} steps, "
            f"need >= {MIN_REASONING_STEPS})"
        )

    # 3. No recommendations
    if len(analysis.recommendations) < MIN_RECOMMENDATIONS:
        escalation_reasons.append("No treatment recommendations provided")

    # 4. Critical safety flags
    critical_flags = [f for f in analysis.safety_flags if f.level == "critical"]
    if critical_flags:
        escalation_reasons.append(
            f"{len(critical_flags)} critical safety flag(s) — mandatory human review"
        )

    # 5. Fallback diagnosis check
    fallback_markers = ["FREE-TEXT", "VALIDATION ERROR", "Unknown", "no JSON"]
    if any(m in analysis.diagnosis.primary for m in fallback_markers):
        escalation_reasons.append(
            "Diagnosis appears to be a fallback/error — LLM output may be malformed"
        )

    # 6. Low evidence strength
    if analysis.confidence.evidence_strength < 0.4:
        escalation_reasons.append(
            f"Weak evidence basis ({analysis.confidence.evidence_strength:.2f})"
        )

    # Final decision
    if escalation_reasons:
        analysis.requires_human_review = True
        new_reason = " | ".join(escalation_reasons)
        if analysis.escalation_reason:
            analysis.escalation_reason = f"{analysis.escalation_reason} | CONFIDENCE: {new_reason}"
        else:
            analysis.escalation_reason = f"CONFIDENCE GATE: {new_reason}"
    else:
        # Only auto-approve if no prior escalation reasons exist
        if not analysis.escalation_reason:
            analysis.requires_human_review = False

    return analysis
