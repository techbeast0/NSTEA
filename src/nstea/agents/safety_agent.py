"""Safety agent — runs hard rule checks on LLM recommendations.

This is a post-processing agent that takes the LLM's AnalysisResponse,
extracts proposed medications, and runs them through the RuleEngine.
It amends the response with any violations found.
"""

from __future__ import annotations

from nstea.models.analysis import AnalysisResponse, SafetyFlag
from nstea.models.patient import PatientInput
from nstea.safety import RuleEngine


_engine: RuleEngine | None = None


def _get_engine() -> RuleEngine:
    global _engine
    if _engine is None:
        _engine = RuleEngine()
    return _engine


def run_safety_check(
    patient: PatientInput, analysis: AnalysisResponse
) -> AnalysisResponse:
    """Post-process an AnalysisResponse through the hard safety rule engine.

    Extracts medication recommendations from the analysis, checks them against
    the patient's allergies, conditions, and current medications, and amends
    the response with any violations.

    Args:
        patient: Original patient input data.
        analysis: LLM-generated analysis response.

    Returns:
        Amended AnalysisResponse with safety flags injected and
        requires_human_review set if critical violations found.
    """
    engine = _get_engine()

    # Extract proposed drugs from recommendations
    proposed_drugs: list[str] = []
    for rec in analysis.recommendations:
        if rec.category == "medication":
            # Extract drug name from the action text
            proposed_drugs.append(rec.action)

    if not proposed_drugs:
        return analysis

    result = engine.check_proposed_drugs(patient, proposed_drugs)

    # Convert rule engine violations to SafetyFlags and collect blocked drug names
    new_flags: list[SafetyFlag] = []
    critical_drugs: set[str] = set()
    for v in result.violations:
        new_flags.append(
            SafetyFlag(
                level=v.severity,
                message=f"[RULE ENGINE] {v.message}",
                related_recommendation=v.blocked_drug,
            )
        )
        if v.severity == "critical":
            if v.blocked_drug:
                critical_drugs.add(v.blocked_drug.lower())
            # For drug interactions (blocked_drug is None), find which proposed
            # drugs are mentioned in the violation message
            elif v.related_rule_type == "drug_interaction":
                msg_lower = v.message.lower()
                for drug_text in proposed_drugs:
                    if drug_text.lower() in msg_lower:
                        critical_drugs.add(drug_text.lower())

    # Remove recommendations that contain critically blocked drugs
    if critical_drugs:
        safe_recs = []
        for rec in analysis.recommendations:
            action_lower = rec.action.lower()
            if any(drug in action_lower for drug in critical_drugs):
                new_flags.append(
                    SafetyFlag(
                        level="critical",
                        message=f"[RULE ENGINE] REMOVED recommendation: '{rec.action}' — contains blocked drug",
                        related_recommendation=rec.action,
                    )
                )
            else:
                safe_recs.append(rec)
        analysis.recommendations = safe_recs

    # Merge with existing flags (avoid duplicates)
    existing_messages = {f.message for f in analysis.safety_flags}
    for flag in new_flags:
        if flag.message not in existing_messages:
            analysis.safety_flags.append(flag)

    # Force human review if critical violations
    if result.has_critical:
        analysis.requires_human_review = True
        reasons = [v.message for v in result.violations if v.severity == "critical"]
        existing_reason = analysis.escalation_reason or ""
        new_reason = "; ".join(reasons)
        if existing_reason:
            analysis.escalation_reason = f"{existing_reason} | RULE ENGINE: {new_reason}"
        else:
            analysis.escalation_reason = f"RULE ENGINE CRITICAL: {new_reason}"

    return analysis
