"""FHIR Bundle loader — converts Synthea FHIR JSON to PatientInput."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from nstea.models.patient import (
    Allergy,
    ClinicalEvent,
    Condition,
    LabResult,
    Medication,
    PatientInput,
    Symptom,
    Vitals,
)


def load_fhir_bundle(path: Path) -> dict:
    """Load a single FHIR Bundle JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_patient_from_bundle(bundle: dict, clinician_query: str = "") -> PatientInput:
    """Convert a FHIR Bundle into a PatientInput model.

    Args:
        bundle: Parsed FHIR Bundle dict.
        clinician_query: Optional query from the clinician.

    Returns:
        PatientInput with extracted clinical data.
    """
    entries = bundle.get("entry", [])
    resources_by_type: dict[str, list[dict]] = {}
    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "Unknown")
        resources_by_type.setdefault(rtype, []).append(resource)

    # Extract patient demographics
    patient_res = resources_by_type.get("Patient", [{}])[0]
    patient_id = patient_res.get("id", "unknown")
    sex = patient_res.get("gender", "other")
    birth_date_str = patient_res.get("birthDate")
    age = _compute_age(birth_date_str) if birth_date_str else 0

    # Extract conditions
    conditions = []
    for res in resources_by_type.get("Condition", []):
        name = _get_display(res.get("code"))
        code = _get_code(res.get("code"))
        onset = _parse_date(res.get("onsetDateTime"))
        status_raw = res.get("clinicalStatus")
        status = _normalize_status(status_raw)
        conditions.append(Condition(
            name=name, snomed_code=code, onset_date=onset, status=status
        ))

    # Extract medications
    medications = []
    for res in resources_by_type.get("MedicationRequest", []):
        med_concept = res.get("medicationCodeableConcept", {})
        name = _get_display(med_concept)
        code = _get_code(med_concept)
        start = _parse_date(res.get("authoredOn"))
        dosage_str = None
        dosage_list = res.get("dosageInstruction", [])
        if dosage_list:
            dosage_str = dosage_list[0].get("text")
        medications.append(Medication(
            name=name, rxnorm_code=code, dosage=dosage_str, start_date=start
        ))

    # Extract allergies
    allergies = []
    for res in resources_by_type.get("AllergyIntolerance", []):
        substance = _get_display(res.get("code"))
        reactions = res.get("reaction", [])
        reaction_str = None
        if reactions:
            manifests = []
            for r in reactions:
                for m in r.get("manifestation", []):
                    manifests.append(_get_display(m))
            reaction_str = "; ".join(manifests) if manifests else None
        criticality = res.get("criticality", "moderate")
        severity_map = {"low": "mild", "high": "severe"}
        severity = severity_map.get(criticality, "moderate")
        allergies.append(Allergy(
            substance=substance, reaction=reaction_str, severity=severity
        ))

    # Extract lab results (Observations with valueQuantity)
    lab_results = []
    for res in resources_by_type.get("Observation", []):
        vq = res.get("valueQuantity")
        if not vq or "value" not in vq:
            continue
        name = _get_display(res.get("code"))
        code = _get_code(res.get("code"))
        obs_date = _parse_date(res.get("effectiveDateTime"))
        if not obs_date:
            continue
        ref_range = None
        ref_list = res.get("referenceRange", [])
        if ref_list:
            low = ref_list[0].get("low", {}).get("value")
            high = ref_list[0].get("high", {}).get("value")
            unit = ref_list[0].get("low", {}).get("unit", "")
            if low is not None and high is not None:
                ref_range = f"{low}-{high} {unit}".strip()
        lab_results.append(LabResult(
            test_name=name,
            loinc_code=code,
            value=float(vq["value"]),
            unit=vq.get("unit", ""),
            reference_range=ref_range,
            date=obs_date,
        ))

    # Extract vitals from most recent encounter observations
    vitals = _extract_latest_vitals(resources_by_type.get("Observation", []))

    # Build chronological history from encounters
    history = []
    for res in resources_by_type.get("Encounter", []):
        period = res.get("period", {})
        enc_date = _parse_date(period.get("start"))
        if enc_date:
            enc_type = "visit"
            desc_parts = []
            for t in res.get("type", []):
                desc_parts.append(_get_display(t))
            desc = "; ".join(desc_parts) if desc_parts else "Encounter"
            history.append(ClinicalEvent(
                event_type=enc_type, description=desc, date=enc_date
            ))

    history.sort(key=lambda e: e.date)

    return PatientInput(
        patient_id=patient_id,
        age=age,
        sex=sex,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        lab_results=lab_results[-20:],  # Most recent 20
        vitals=vitals,
        history=history[-15:],  # Most recent 15 events
        clinician_query=clinician_query,
    )


def list_fhir_files(data_dir: Path, limit: Optional[int] = None) -> list[Path]:
    """List all FHIR JSON files in the hash-bucketed directory."""
    files = sorted(data_dir.rglob("*.json"))
    if limit:
        files = files[:limit]
    return files


# --- Helpers ---

def _get_display(codeable_concept: Optional[dict]) -> str:
    if not codeable_concept or not isinstance(codeable_concept, dict):
        return "Unknown"
    codings = codeable_concept.get("coding", [])
    if codings and isinstance(codings[0], dict):
        return codings[0].get("display", codeable_concept.get("text", "Unknown"))
    return codeable_concept.get("text", "Unknown")


def _get_code(codeable_concept: Optional[dict]) -> Optional[str]:
    if not codeable_concept or not isinstance(codeable_concept, dict):
        return None
    codings = codeable_concept.get("coding", [])
    if codings and isinstance(codings[0], dict):
        return codings[0].get("code")
    return None


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        try:
            return date.fromisoformat(date_str[:10])
        except (ValueError, AttributeError):
            return None


def _compute_age(birth_date_str: str) -> int:
    try:
        bd = date.fromisoformat(birth_date_str[:10])
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except (ValueError, AttributeError):
        return 0


def _normalize_status(status_raw) -> str:
    if isinstance(status_raw, str):
        s = status_raw.lower()
    elif isinstance(status_raw, dict):
        codings = status_raw.get("coding", [{}])
        s = codings[0].get("code", "active").lower() if codings else "active"
    else:
        s = "active"
    if s in ("active",):
        return "active"
    if s in ("resolved", "inactive", "remission"):
        return "resolved"
    if s in ("recurrence", "relapse"):
        return "chronic"
    return "active"


VITAL_MAP = {
    "heart rate": "heart_rate",
    "respiratory rate": "respiratory_rate",
    "body temperature": "temperature",
    "oxygen saturation": "spo2",
    "body mass index": None,  # skip
    "body weight": None,
    "body height": None,
}


def _extract_latest_vitals(observations: list[dict]) -> Optional[Vitals]:
    """Extract the most recent set of vitals from observations."""
    vital_obs = []
    for obs in observations:
        display = _get_display(obs.get("code", {})).lower()
        vq = obs.get("valueQuantity")
        dt = obs.get("effectiveDateTime")
        if not vq or not dt:
            continue
        if any(k in display for k in ["blood pressure", "heart rate", "temperature",
                                        "respiratory", "oxygen saturation"]):
            vital_obs.append((dt, display, vq))

    if not vital_obs:
        return None

    # Sort by date descending and take most recent
    vital_obs.sort(key=lambda x: x[0], reverse=True)

    hr = bp_sys = bp_dia = temp = rr = spo2 = None
    for dt, display, vq in vital_obs:
        val = vq.get("value")
        if val is None:
            continue
        if "heart rate" in display and hr is None:
            hr = int(val)
        elif "systolic" in display and bp_sys is None:
            bp_sys = int(val)
        elif "diastolic" in display and bp_dia is None:
            bp_dia = int(val)
        elif "temperature" in display and temp is None:
            temp = float(val)
        elif "respiratory" in display and rr is None:
            rr = int(val)
        elif "oxygen" in display and spo2 is None:
            spo2 = float(val)

    if any([hr, bp_sys, temp, rr, spo2]):
        return Vitals(
            heart_rate=hr,
            blood_pressure_systolic=bp_sys,
            blood_pressure_diastolic=bp_dia,
            temperature=temp,
            respiratory_rate=rr,
            spo2=spo2,
        )
    return None
