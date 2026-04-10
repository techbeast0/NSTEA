"""Drug safety tool: check drug-drug interactions and allergy contraindications.

Wraps the RuleEngine as a callable function for agent pipelines.
"""

from __future__ import annotations

from nstea.models.patient import PatientInput
from nstea.safety import RuleEngine

_engine: RuleEngine | None = None


def _get_engine() -> RuleEngine:
    global _engine
    if _engine is None:
        _engine = RuleEngine()
    return _engine


def check_drug_safety(patient_json: str, proposed_drugs_csv: str) -> str:
    """Check proposed drugs for safety issues against patient data.

    This tool checks:
    - Direct allergy matches (e.g. prescribing a drug the patient is allergic to)
    - Cross-reactivity (e.g. penicillin allergy → cephalosporin caution)
    - Condition-drug contraindications (e.g. pregnancy + isotretinoin)
    - Drug-drug interactions with current medications

    Args:
        patient_json: JSON string of the patient data (PatientInput schema).
        proposed_drugs_csv: Comma-separated list of proposed drug names.

    Returns:
        A text summary of safety results — violations, warnings, and a safe/unsafe verdict.
    """
    import json

    try:
        data = json.loads(patient_json)
        patient = PatientInput(**data)
    except Exception as e:
        return f"ERROR: Could not parse patient data — {e}"

    proposed = [d.strip() for d in proposed_drugs_csv.split(",") if d.strip()]
    if not proposed:
        return "No drugs proposed to check."

    engine = _get_engine()
    result = engine.check_proposed_drugs(patient, proposed)

    lines = []
    if result.is_safe and not result.violations:
        lines.append("✅ SAFE: No safety violations found for proposed drugs.")
    elif result.is_safe:
        lines.append("⚠️ WARNINGS FOUND (no critical violations):")
    else:
        lines.append("🚫 UNSAFE: Critical violations detected!")

    for v in result.violations:
        icon = "🚫" if v.severity == "critical" else "⚠️" if v.severity == "warning" else "ℹ️"
        lines.append(f"  {icon} [{v.rule_id}] {v.message}")
        if v.blocked_drug:
            lines.append(f"      → Blocked drug: {v.blocked_drug}")

    return "\n".join(lines)
