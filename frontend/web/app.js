// NS-TEA Frontend Application Logic
// Alpine.js component

function nstea() {
  return {
    // Navigation
    page: 'dashboard',

    // State
    health: { status: 'unknown', version: '', phase: '' },
    feedbackSummary: {},
    analyses: [],
    currentResult: null,
    analyzing: false,
    analyzeError: '',
    showImportModal: false,
    importText: '',
    importError: '',

    // Patient form
    form: defaultForm(),

    // Lifecycle
    async init() {
      await this.refreshHealth();
      await this.refreshFeedback();
    },

    // ── API Helpers ───────────────────────────────────
    async api(method, path, body) {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch('/api/v1' + path, opts);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
      }
      return res.json();
    },

    async refreshHealth() {
      try {
        const r = await fetch('/health');
        if (r.ok) {
          this.health = await r.json();
        } else {
          this.health = { status: 'error', version: '?', phase: '?' };
        }
      } catch {
        this.health = { status: 'offline', version: '?', phase: '?' };
      }
    },

    async refreshFeedback() {
      try {
        const r = await this.api('GET', '/feedback-summary');
        this.feedbackSummary = r;
        this.$nextTick(() => this.renderFeedbackChart());
      } catch { /* ignore */ }
    },

    renderFeedbackChart() {
      const el = document.getElementById('feedbackChart');
      if (!el || !this.feedbackSummary.total) return;
      // Destroy existing chart
      const existing = Chart.getChart(el);
      if (existing) existing.destroy();

      new Chart(el, {
        type: 'doughnut',
        data: {
          labels: ['Accepted', 'Modified', 'Rejected'],
          datasets: [{
            data: [
              this.feedbackSummary.accepted || 0,
              this.feedbackSummary.modified || 0,
              this.feedbackSummary.rejected || 0,
            ],
            backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true,
          plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 12 } } },
          cutout: '60%',
        },
      });
    },

    // ── Analysis ──────────────────────────────────────
    async runAnalysis() {
      this.analyzing = true;
      this.analyzeError = '';
      try {
        // Build the payload, filtering out empty entries
        const payload = buildPayload(this.form);
        const result = await this.api('POST', '/analyze', payload);
        this.currentResult = result;
        this.analyses.unshift(result);
        this.page = 'results';
      } catch (e) {
        this.analyzeError = 'Analysis failed: ' + e.message;
      } finally {
        this.analyzing = false;
      }
    },

    // ── Feedback ──────────────────────────────────────
    async submitFeedback(analysisId, verdict, notes, modDx) {
      await this.api('POST', '/feedback', {
        analysis_id: analysisId,
        clinician_id: 'web-user',
        verdict: verdict,
        notes: notes || '',
        modified_diagnosis: modDx || null,
      });
      await this.refreshFeedback();
    },

    // ── Templates ─────────────────────────────────────
    loadTemplate(name) {
      this.form = TEMPLATES[name] ? { ...defaultForm(), ...JSON.parse(JSON.stringify(TEMPLATES[name])) } : defaultForm();
    },

    resetForm() {
      this.form = defaultForm();
    },

    importJSON() {
      this.importText = '';
      this.importError = '';
      this.showImportModal = true;
    },

    doImport() {
      try {
        const data = JSON.parse(this.importText);
        this.form = {
          patient_id: data.patient_id || 'import_001',
          age: data.age || 50,
          sex: data.sex || 'male',
          conditions: data.conditions || [],
          medications: data.medications || [],
          allergies: data.allergies || [],
          symptoms: data.symptoms || [],
          lab_results: (data.lab_results || []).map(l => ({
            ...l,
            value: Number(l.value) || 0,
          })),
          vitals: data.vitals || { heart_rate: null, blood_pressure_systolic: null, blood_pressure_diastolic: null, temperature: null, respiratory_rate: null, spo2: null },
          history: data.history || [],
          clinician_query: data.clinician_query || '',
        };
        this.showImportModal = false;
      } catch (e) {
        this.importError = 'Invalid JSON: ' + e.message;
      }
    },
  };
}

// ── Calculator API ────────────────────────────────────
async function calcAPI(name, body) {
  const res = await fetch('/api/v1/calculators/' + name, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Calculation failed');
  }
  return res.json();
}

// ── Utilities ─────────────────────────────────────────
function today() {
  return new Date().toISOString().split('T')[0];
}

function defaultForm() {
  return {
    patient_id: 'P-' + Math.random().toString(36).substring(2, 7).toUpperCase(),
    age: 65,
    sex: 'male',
    conditions: [],
    medications: [],
    allergies: [],
    symptoms: [],
    lab_results: [],
    vitals: {
      heart_rate: null,
      blood_pressure_systolic: null,
      blood_pressure_diastolic: null,
      temperature: null,
      respiratory_rate: null,
      spo2: null,
    },
    history: [],
    clinician_query: '',
  };
}

function buildPayload(form) {
  const f = JSON.parse(JSON.stringify(form));
  // Filter empty entries
  f.conditions = f.conditions.filter(c => c.name.trim());
  f.medications = f.medications.filter(m => m.name.trim());
  f.allergies = f.allergies.filter(a => a.substance.trim());
  f.symptoms = f.symptoms.filter(s => s.description.trim());
  f.lab_results = f.lab_results.filter(l => l.test_name.trim());
  f.history = f.history.filter(h => h.description.trim());

  // Clean vitals — remove nulls/zeros
  const v = f.vitals;
  const hasVitals = Object.values(v).some(x => x !== null && x !== 0 && x !== '');
  if (!hasVitals) {
    f.vitals = null;
  }

  return f;
}

// ── Quick-fill templates ──────────────────────────────
const TEMPLATES = {
  cardiac: {
    patient_id: 'CARDIAC-001',
    age: 65,
    sex: 'male',
    conditions: [
      { name: 'Hypertension', icd10_code: 'I10', status: 'chronic' },
      { name: 'Type 2 Diabetes', icd10_code: 'E11.9', status: 'active' },
    ],
    medications: [
      { name: 'Metformin', dosage: '500mg BID' },
      { name: 'Lisinopril', dosage: '10mg QD' },
    ],
    allergies: [
      { substance: 'Aspirin', reaction: 'anaphylaxis', severity: 'severe' },
    ],
    symptoms: [
      { description: 'Chest pain, acute onset 2 hours ago', severity: 'severe' },
      { description: 'Shortness of breath', severity: 'moderate' },
    ],
    lab_results: [
      { test_name: 'Troponin I', value: 0.8, unit: 'ng/mL', date: today(), reference_range: '<0.04', is_abnormal: true },
      { test_name: 'HbA1c', value: 7.2, unit: '%', date: today(), reference_range: '4-5.6%', is_abnormal: true },
    ],
    vitals: {
      heart_rate: 98,
      blood_pressure_systolic: 155,
      blood_pressure_diastolic: 92,
      temperature: null,
      respiratory_rate: 22,
      spo2: 95,
    },
    history: [
      { event_type: 'medication', description: 'Started NSAIDs for joint pain', date: '2025-10-15' },
      { event_type: 'diagnosis', description: 'Diagnosed Type 2 Diabetes', date: '2025-06-20' },
      { event_type: 'diagnosis', description: 'Hypertension diagnosed', date: '2024-11-01' },
    ],
    clinician_query: 'What is the likely diagnosis and recommended treatment plan? Patient has aspirin allergy.',
  },

  diabetes: {
    patient_id: 'DM-002',
    age: 58,
    sex: 'female',
    conditions: [
      { name: 'Type 2 Diabetes', icd10_code: 'E11.9', status: 'active' },
      { name: 'Hypertension', icd10_code: 'I10', status: 'chronic' },
      { name: 'Chronic Kidney Disease', icd10_code: 'N18.3', status: 'active' },
    ],
    medications: [
      { name: 'Metformin', dosage: '1000mg BID' },
      { name: 'Losartan', dosage: '50mg QD' },
      { name: 'Amlodipine', dosage: '5mg QD' },
    ],
    allergies: [
      { substance: 'Penicillin', reaction: 'Rash', severity: 'moderate' },
    ],
    symptoms: [
      { description: 'Persistent headache for 3 days', severity: 'moderate' },
      { description: 'Blurry vision', severity: 'mild' },
      { description: 'Fatigue', severity: 'moderate' },
    ],
    lab_results: [
      { test_name: 'HbA1c', value: 8.5, unit: '%', date: today(), reference_range: '4-5.6%', is_abnormal: true },
      { test_name: 'Creatinine', value: 1.8, unit: 'mg/dL', date: today(), reference_range: '0.6-1.1', is_abnormal: true },
      { test_name: 'Fasting glucose', value: 195, unit: 'mg/dL', date: today(), reference_range: '70-100', is_abnormal: true },
    ],
    vitals: {
      heart_rate: 82,
      blood_pressure_systolic: 165,
      blood_pressure_diastolic: 98,
      temperature: 36.8,
      respiratory_rate: 16,
      spo2: 98,
    },
    history: [],
    clinician_query: 'Patient has uncontrolled diabetes with CKD Stage 3. Assess medication safety and recommend adjustments.',
  },

  respiratory: {
    patient_id: 'RESP-003',
    age: 71,
    sex: 'male',
    conditions: [
      { name: 'COPD', icd10_code: 'J44.1', status: 'chronic' },
      { name: 'Atrial Fibrillation', icd10_code: 'I48.91', status: 'active' },
      { name: 'Heart Failure', icd10_code: 'I50.9', status: 'active' },
    ],
    medications: [
      { name: 'Warfarin', dosage: '5mg QD' },
      { name: 'Digoxin', dosage: '0.125mg QD' },
      { name: 'Furosemide', dosage: '40mg QD' },
    ],
    allergies: [],
    symptoms: [
      { description: 'Worsening dyspnea over 3 days', severity: 'severe' },
      { description: 'Productive cough with yellow sputum', severity: 'moderate' },
      { description: 'Lower extremity edema', severity: 'moderate' },
    ],
    lab_results: [
      { test_name: 'BNP', value: 890, unit: 'pg/mL', date: today(), reference_range: '<100', is_abnormal: true },
      { test_name: 'INR', value: 2.8, unit: '', date: today(), reference_range: '2.0-3.0', is_abnormal: false },
    ],
    vitals: {
      heart_rate: 110,
      blood_pressure_systolic: 135,
      blood_pressure_diastolic: 85,
      temperature: 37.8,
      respiratory_rate: 28,
      spo2: 89,
    },
    history: [
      { event_type: 'diagnosis', description: 'COPD exacerbation requiring hospitalization', date: '2025-11-10' },
      { event_type: 'diagnosis', description: 'Atrial fibrillation diagnosed, started warfarin', date: '2025-03-15' },
    ],
    clinician_query: 'Acute COPD exacerbation vs heart failure decompensation? Treatment recommendations considering drug interactions.',
  },

  safety: {
    patient_id: 'SAFETY-001',
    age: 45,
    sex: 'female',
    conditions: [
      { name: 'Asthma', icd10_code: 'J45', status: 'active' },
      { name: 'GI Bleeding', icd10_code: 'K92.2', status: 'active' },
    ],
    medications: [
      { name: 'Warfarin', dosage: '5mg QD' },
      { name: 'SSRI', dosage: '20mg QD' },
    ],
    allergies: [
      { substance: 'Aspirin', reaction: 'anaphylaxis', severity: 'severe' },
      { substance: 'Penicillin', reaction: 'Hives', severity: 'moderate' },
    ],
    symptoms: [
      { description: 'Knee pain from sports injury', severity: 'moderate' },
      { description: 'Anxiety worsening', severity: 'mild' },
    ],
    lab_results: [],
    vitals: {
      heart_rate: 76,
      blood_pressure_systolic: 120,
      blood_pressure_diastolic: 78,
      temperature: null,
      respiratory_rate: null,
      spo2: null,
    },
    history: [],
    clinician_query: 'Patient needs pain management for knee. What are safe options given aspirin allergy, GI bleeding, asthma, and current warfarin + SSRI?',
  },
};
