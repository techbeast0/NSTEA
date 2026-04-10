"""Patient and clinical data models for NS-TEA."""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Condition(BaseModel):
    name: str
    icd10_code: Optional[str] = None
    snomed_code: Optional[str] = None
    onset_date: Optional[date] = None
    status: Literal["active", "resolved", "chronic"] = "active"


class Medication(BaseModel):
    name: str
    rxnorm_code: Optional[str] = None
    dosage: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class Allergy(BaseModel):
    substance: str
    reaction: Optional[str] = None
    severity: Literal["mild", "moderate", "severe"] = "moderate"


class Symptom(BaseModel):
    description: str
    onset: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None


class LabResult(BaseModel):
    test_name: str
    loinc_code: Optional[str] = None
    value: float
    unit: str
    reference_range: Optional[str] = None
    date: date
    is_abnormal: Optional[bool] = None


class Vitals(BaseModel):
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[int] = None
    spo2: Optional[float] = None


class ClinicalEvent(BaseModel):
    event_type: Literal["diagnosis", "medication", "procedure", "lab", "visit", "imaging"]
    description: str
    date: date
    details: Optional[dict] = None


class PatientInput(BaseModel):
    """What the clinician submits for analysis."""

    patient_id: str
    age: int
    sex: Literal["male", "female", "other"]
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    symptoms: list[Symptom] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)
    vitals: Optional[Vitals] = None
    history: list[ClinicalEvent] = Field(default_factory=list)
    clinician_query: str = ""

    def to_clinical_summary(self) -> str:
        """Convert patient data to a structured text summary for LLM consumption."""
        lines = [
            f"PATIENT: {self.patient_id} | Age: {self.age} | Sex: {self.sex}",
            "",
        ]

        if self.conditions:
            lines.append("ACTIVE CONDITIONS:")
            for c in self.conditions:
                code = c.icd10_code or c.snomed_code or "no code"
                lines.append(f"  - {c.name} ({code}) [{c.status}]")
            lines.append("")

        if self.medications:
            lines.append("CURRENT MEDICATIONS:")
            for m in self.medications:
                dose = f" {m.dosage}" if m.dosage else ""
                lines.append(f"  - {m.name}{dose}")
            lines.append("")

        if self.allergies:
            lines.append("KNOWN ALLERGIES:")
            for a in self.allergies:
                reaction = f" → {a.reaction}" if a.reaction else ""
                lines.append(f"  - {a.substance}{reaction} [{a.severity}]")
            lines.append("")

        if self.symptoms:
            lines.append("PRESENTING SYMPTOMS:")
            for s in self.symptoms:
                sev = f" [{s.severity}]" if s.severity else ""
                lines.append(f"  - {s.description}{sev}")
            lines.append("")

        if self.lab_results:
            lines.append("RECENT LAB RESULTS:")
            for lab in sorted(self.lab_results, key=lambda x: x.date, reverse=True)[:15]:
                abnormal = " ⚠️ ABNORMAL" if lab.is_abnormal else ""
                ref = f" (ref: {lab.reference_range})" if lab.reference_range else ""
                lines.append(
                    f"  - [{lab.date}] {lab.test_name}: {lab.value} {lab.unit}{ref}{abnormal}"
                )
            lines.append("")

        if self.vitals:
            v = self.vitals
            parts = []
            if v.heart_rate:
                parts.append(f"HR {v.heart_rate}")
            if v.blood_pressure_systolic and v.blood_pressure_diastolic:
                parts.append(f"BP {v.blood_pressure_systolic}/{v.blood_pressure_diastolic}")
            if v.temperature:
                parts.append(f"Temp {v.temperature}°C")
            if v.respiratory_rate:
                parts.append(f"RR {v.respiratory_rate}")
            if v.spo2:
                parts.append(f"SpO2 {v.spo2}%")
            if parts:
                lines.append(f"VITALS: {' | '.join(parts)}")
                lines.append("")

        if self.clinician_query:
            lines.append(f"CLINICIAN QUERY: {self.clinician_query}")

        return "\n".join(lines)
