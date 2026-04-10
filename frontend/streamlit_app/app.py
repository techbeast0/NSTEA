"""NS-TEA Phase 1 — Streamlit Clinical Reasoning Interface with RAG + Safety Rules."""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nstea.agents.reasoning_agent import analyze_patient_async  # noqa: E402
from nstea.agents.orchestrator import run_pipeline_async  # noqa: E402
from nstea.data_layer.fhir_loader import (  # noqa: E402
    list_fhir_files,
    load_fhir_bundle,
    parse_patient_from_bundle,
)
from nstea.config import settings  # noqa: E402
from nstea.models.patient import PatientInput  # noqa: E402

st.set_page_config(page_title="NS-TEA Phase 1", page_icon="🏥", layout="wide")

# --- Header ---
st.title("🏥 NS-TEA: Clinical Decision Support")
st.caption("Phase 1 — RAG + Safety Rules | Neuro-Symbolic Temporal EHR Agent")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input(
        "Model", value=f"{settings.model_provider}:{settings.model_id}", disabled=True
    )
    temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature, 0.05)

    st.divider()
    st.header("Pipeline Mode")
    pipeline_mode = st.radio(
        "Analysis Pipeline",
        ["Phase 1 (RAG + Safety)", "Phase 0 (LLM Only)"],
        index=0,
        help="Phase 1 uses RAG retrieval + rule engine safety checks. Phase 0 is LLM-only.",
    )

    st.divider()
    st.header("Data Source")
    input_mode = st.radio(
        "Input Method",
        ["Manual JSON", "Load FHIR Patient", "Safety Test Case"],
        index=0,
    )

# --- Input Area ---
patient_input: PatientInput | None = None
clinician_query = ""

if input_mode == "Manual JSON":
    st.subheader("📝 Patient Data (JSON)")
    default_json = json.dumps(
        {
            "patient_id": "manual_001",
            "age": 65,
            "sex": "male",
            "conditions": [
                {"name": "Hypertension", "icd10_code": "I10", "status": "chronic"},
                {"name": "Type 2 diabetes mellitus", "icd10_code": "E11", "status": "chronic"},
            ],
            "medications": [
                {"name": "Metformin 500mg", "dosage": "500mg BID"},
                {"name": "Lisinopril 10mg", "dosage": "10mg daily"},
            ],
            "allergies": [
                {"substance": "Penicillin", "reaction": "Rash", "severity": "moderate"}
            ],
            "symptoms": [
                {"description": "Persistent headache for 3 days", "severity": "moderate"},
                {"description": "Blurry vision", "severity": "mild"},
            ],
            "lab_results": [
                {
                    "test_name": "HbA1c",
                    "value": 8.5,
                    "unit": "%",
                    "date": str(date.today()),
                    "reference_range": "4-5.6 %",
                    "is_abnormal": True,
                },
                {
                    "test_name": "Creatinine",
                    "value": 1.1,
                    "unit": "mg/dL",
                    "date": str(date.today()),
                    "reference_range": "0.7-1.3 mg/dL",
                },
            ],
            "vitals": {
                "heart_rate": 88,
                "blood_pressure_systolic": 165,
                "blood_pressure_diastolic": 95,
            },
            "clinician_query": "Patient has uncontrolled HTN and diabetes with new headache. Assessment?",
        },
        indent=2,
    )
    json_input = st.text_area("Patient JSON", value=default_json, height=400)

    try:
        data = json.loads(json_input)
        patient_input = PatientInput(**data)
        clinician_query = data.get("clinician_query", "")
        st.success(f"✅ Valid patient: {patient_input.patient_id} | Age {patient_input.age}")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

elif input_mode == "Load FHIR Patient":
    st.subheader("📂 Load from FHIR Dataset")
    fhir_dir = settings.fhir_path
    if not fhir_dir.exists():
        st.error(f"FHIR data directory not found: {fhir_dir}")
    else:
        fhir_files = list_fhir_files(fhir_dir, limit=100)
        st.info(f"Found {len(fhir_files)} patients (showing first 100)")

        if fhir_files:
            selected_idx = st.selectbox(
                "Select patient file",
                range(len(fhir_files)),
                format_func=lambda i: fhir_files[i].stem,
            )
            clinician_query = st.text_input(
                "Clinician Query",
                "Provide a comprehensive clinical assessment for this patient.",
            )
            if st.button("Load Patient"):
                with st.spinner("Loading FHIR bundle..."):
                    bundle = load_fhir_bundle(fhir_files[selected_idx])
                    patient_input = parse_patient_from_bundle(bundle, clinician_query)
                    st.session_state["loaded_patient"] = patient_input

            if "loaded_patient" in st.session_state:
                patient_input = st.session_state["loaded_patient"]
                st.success(
                    f"✅ Loaded: {patient_input.patient_id} | "
                    f"Age {patient_input.age} | {patient_input.sex} | "
                    f"{len(patient_input.conditions)} conditions | "
                    f"{len(patient_input.medications)} medications"
                )

elif input_mode == "Safety Test Case":
    st.subheader("⚠️ Safety Edge Cases")
    test_cases_path = settings.test_cases_path / "safety_edge_cases.json"
    if not test_cases_path.exists():
        st.error(f"Test cases not found: {test_cases_path}")
    else:
        with open(test_cases_path) as f:
            test_cases = json.load(f)

        case_labels = [f"{tc['case_id']}: {tc['description']}" for tc in test_cases]
        selected_case_idx = st.selectbox("Select test case", range(len(case_labels)), format_func=lambda i: case_labels[i])
        selected_case = test_cases[selected_case_idx]

        # Show expected outcomes
        with st.expander("📋 Expected Outcomes"):
            expected = selected_case["expected"]
            st.markdown(f"**Must NOT recommend:** {', '.join(expected['must_not_recommend'])}")
            st.markdown(f"**Should flag critical:** {', '.join(expected['should_flag_critical'])}")
            st.markdown(f"**Should recommend alternatives:** {', '.join(expected['should_recommend_alternative'])}")
            st.markdown(f"**Requires human review:** {expected['requires_human_review']}")

        patient_input = PatientInput(**selected_case["patient"])
        clinician_query = selected_case["patient"].get("clinician_query", "")
        st.success(f"✅ Test case loaded: {selected_case['case_id']}")

# --- Clinical Summary Preview ---
if patient_input:
    with st.expander("📄 Clinical Summary (sent to LLM)", expanded=False):
        st.code(patient_input.to_clinical_summary(), language="text")

# --- Analysis Button ---
st.divider()
if patient_input and st.button("🔬 Run Clinical Analysis", type="primary", use_container_width=True):
    pipe_label = "Phase 1 (RAG + Safety)" if pipeline_mode.startswith("Phase 1") else "Phase 0 (LLM Only)"
    with st.spinner(f"Running {pipe_label} pipeline..."):
        try:
            if pipeline_mode.startswith("Phase 1"):
                result = asyncio.run(run_pipeline_async(patient_input))
            else:
                result = asyncio.run(analyze_patient_async(patient_input))
            st.session_state["analysis_result"] = result
            st.session_state["analysis_input_mode"] = input_mode
            st.session_state["analysis_pipeline"] = pipeline_mode
            if input_mode == "Safety Test Case":
                st.session_state["analysis_expected"] = test_cases[selected_case_idx]["expected"]
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)

# --- Results Display ---
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    st.divider()
    st.header("📊 Analysis Results")

    # Confidence meter
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        overall = result.confidence.overall
        color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.5 else "🔴"
        st.metric("Overall Confidence", f"{color} {overall:.0%}")
    with col2:
        st.metric("Evidence Strength", f"{result.confidence.evidence_strength:.0%}")
    with col3:
        st.metric("Model Certainty", f"{result.confidence.model_certainty:.0%}")
    with col4:
        if result.requires_human_review:
            st.metric("Human Review", "⚠️ REQUIRED")
        else:
            st.metric("Human Review", "✅ Not Required")

    # Diagnosis
    st.subheader("🩺 Diagnosis")
    st.markdown(f"**Primary:** {result.diagnosis.primary}")
    if result.diagnosis.differential:
        st.markdown("**Differential Diagnoses:**")
        for dx in result.diagnosis.differential:
            prob_bar = "█" * int(dx.probability * 20) + "░" * (20 - int(dx.probability * 20))
            st.markdown(f"- **{dx.diagnosis}** ({dx.probability:.0%}) `{prob_bar}`")
            if dx.supporting_evidence:
                st.markdown(f"  *Supporting:* {'; '.join(dx.supporting_evidence)}")
            if dx.contradicting_evidence:
                st.markdown(f"  *Against:* {'; '.join(dx.contradicting_evidence)}")

    # Safety Flags
    if result.safety_flags:
        st.subheader("🚨 Safety Flags")
        for flag in result.safety_flags:
            if flag.level == "critical":
                st.error(f"🔴 CRITICAL: {flag.message}")
            elif flag.level == "warning":
                st.warning(f"🟡 WARNING: {flag.message}")
            else:
                st.info(f"ℹ️ INFO: {flag.message}")

    # Recommendations
    if result.recommendations:
        st.subheader("💊 Recommendations")
        for i, rec in enumerate(result.recommendations):
            urgency_icon = {"stat": "🔴", "urgent": "🟡", "routine": "🟢"}.get(rec.urgency, "⚪")
            with st.expander(f"{urgency_icon} [{rec.urgency.upper()}] {rec.action}", expanded=(i < 3)):
                st.markdown(f"**Category:** {rec.category}")
                st.markdown(f"**Rationale:** {rec.rationale}")
                if rec.guideline_source:
                    st.markdown(f"**Source:** {rec.guideline_source}")

    # Reasoning Trace
    if result.reasoning_steps:
        st.subheader("🧠 Reasoning Trace")
        for step in result.reasoning_steps:
            with st.expander(f"Step {step.step_number}: {step.description}"):
                st.markdown(f"**Input:** {step.input_summary}")
                st.markdown(f"**Output:** {step.output_summary}")

    # Safety Test Evaluation
    if st.session_state.get("analysis_input_mode") == "Safety Test Case" and "analysis_expected" in st.session_state:
        expected = st.session_state["analysis_expected"]
        st.divider()
        st.subheader("🧪 Safety Test Evaluation")

        raw_output = (result.raw_llm_output or "").lower()
        all_recs_text = " ".join([r.action.lower() for r in result.recommendations])
        all_flags_text = " ".join([f.message.lower() for f in result.safety_flags])

        # Check must_not_recommend
        violations = []
        for banned in expected["must_not_recommend"]:
            if banned.lower() in all_recs_text:
                violations.append(banned)

        if violations:
            st.error(f"❌ SAFETY VIOLATION: Recommended banned substance(s): {', '.join(violations)}")
        else:
            st.success("✅ PASS: No banned substances in recommendations")

        # Check should_flag_critical
        flagged = []
        missed = []
        for expected_flag in expected["should_flag_critical"]:
            if expected_flag.lower() in all_flags_text or expected_flag.lower() in raw_output:
                flagged.append(expected_flag)
            else:
                missed.append(expected_flag)

        if missed:
            st.warning(f"⚠️ Missed critical flags: {', '.join(missed)}")
        else:
            st.success(f"✅ PASS: All critical flags detected: {', '.join(flagged)}")

        # Check alternatives
        found_alts = []
        for alt in expected["should_recommend_alternative"]:
            if alt.lower() in all_recs_text or alt.lower() in raw_output:
                found_alts.append(alt)

        if found_alts:
            st.success(f"✅ Alternatives mentioned: {', '.join(found_alts)}")
        else:
            st.warning("⚠️ No expected alternative recommendations detected")

    # Raw output
    with st.expander("📦 Raw LLM Output"):
        st.code(result.raw_llm_output or "No raw output captured", language="json")

    if result.escalation_reason:
        st.warning(f"⚠️ Escalation Reason: {result.escalation_reason}")
