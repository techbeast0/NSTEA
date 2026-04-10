# NS-TEA: Neuro-Symbolic Temporal EHR Agent
## Production-Grade Implementation Plan

---

> **Document Version**: 1.0  
> **Date**: April 7, 2026  
> **Status**: AWAITING APPROVAL — No code will be written until this plan is approved.  
> **Language**: Python  
> **Agentic Framework**: Agno Agent Framework

---

## Table of Contents

1. [System Overview & North Star Architecture](#1-system-overview--north-star-architecture)
2. [Complete Tech Stack](#2-complete-tech-stack)
3. [Repository Structure](#3-repository-structure)
4. [Data Models & Schemas](#4-data-models--schemas)
5. [Backend Architecture (Detailed)](#5-backend-architecture-detailed)
6. [Frontend UI/UX Design](#6-frontend-uiux-design)
7. [Agno Agent Architecture](#7-agno-agent-architecture)
8. [API Contracts](#8-api-contracts)
9. [Phase 0: Problem Validation](#9-phase-0-problem-validation)
10. [Phase 1: MVP (RAG + LLM + Rule Engine)](#10-phase-1-mvp)
11. [Phase 2: Production MVP (Systemization)](#11-phase-2-production-mvp)
12. [Phase 3: Structured Reasoning Upgrade](#12-phase-3-structured-reasoning-upgrade)
13. [Phase 4: Temporal Layer (T-GNN Lite)](#13-phase-4-temporal-layer)
14. [Phase 5: Symbolic Constraint Engine](#14-phase-5-symbolic-constraint-engine)
15. [Phase 6: System Hardening](#15-phase-6-system-hardening)
16. [Phase 7: Advanced Features](#16-phase-7-advanced-features)
17. [Testing Strategy](#17-testing-strategy)
18. [Deployment Strategy](#18-deployment-strategy)
19. [Security & Compliance](#19-security--compliance)
20. [Risk Matrix & Mitigations](#20-risk-matrix--mitigations)
21. [Timeline & Milestones](#21-timeline--milestones)

---

## 1. System Overview & North Star Architecture

### 1.1 Problem Statement

Clinical decision-making suffers from:
- **Fragmented Information**: Patient history spread across EHR, notes, labs, imaging
- **Cognitive Overload**: Physicians must recall guidelines, interpret data, decide quickly
- **Temporal Blindness**: Current AI treats patient history as flat text, missing time-causal relationships
- **Unsafe AI**: LLMs hallucinate and give unsafe suggestions without guardrails
- **No Explainability**: "Why this decision?" remains unanswerable

### 1.2 Solution

NS-TEA is a **closed-loop clinical decision support system**:

```
Patient Data (FHIR/Manual)
         ↓
┌─────────────────────────────────────────────────────────┐
│  PREPROCESSING: Standardize → ICD-10, RxNorm, LOINC     │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  TEMPORAL ENGINE (T-GNN Lite)                            │
│  patient history → temporal graph → dense embedding      │
│  importance ∝ e^(-λ * time_gap)                          │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  RETRIEVAL (RAG)                                         │
│  patient_embedding + symptoms + query → Vector DB        │
│  → Top-K clinical guidelines + protocols                 │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  REASONING (LLM Orchestrator via Agno)                    │
│  Decompose → Hypothesize → Evidence Match → Propose      │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  SAFETY (Symbolic Constraint Engine)                     │
│  Condition(patient) ∩ Contraindication(action) = ∅       │
│  IF violation → REJECT → force LLM re-generation         │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  CONFIDENCE GATE                                         │
│  if confidence < τ → escalate to human                   │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  HUMAN-IN-THE-LOOP                                       │
│  Clinician: Accept / Modify / Reject                     │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  FEEDBACK LOOP                                           │
│  Store: input + output + correction → retrain/evaluate    │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Modular Isolation** | Each layer is an independent service; failure in one does not cascade |
| **Safe-by-Design** | Constraint engine + HITL mandatory; no direct EHR write-access |
| **Grounded** | Every recommendation traceable to retrieved evidence (VLY metric) |
| **Adaptive** | Feedback loop enables continuous improvement |
| **Read-Only** | System NEVER writes to EHR; only generates recommendations |
| **Latency Budget** | < 5 seconds for simple cases, < 15 seconds for complex cases |

---

## 2. Complete Tech Stack

### 2.1 Backend Stack

| Layer | Technology | Purpose | Version |
|-------|-----------|---------|---------|
| **Language** | Python | Primary development language | 3.11+ |
| **Web Framework** | FastAPI | High-performance async REST API | 0.115+ |
| **Agentic Framework** | Agno (Python) | Multi-agent orchestration, workflows, sessions | Latest |
| **LLM Provider** | Ollama (local) / HuggingFace (cloud) | Core reasoning engine | gpt-oss:20b (Ollama) / HuggingFace models |
| **Embeddings** | BioClinicalBERT / text-embedding-004 | Medical text embeddings | - |
| **Vector Database** | Qdrant | Semantic search for clinical guidelines | 1.9+ |
| **Relational Database** | PostgreSQL | Patient records, audit logs, user data | 16+ |
| **Graph Database** | Neo4j | Knowledge graph (SNOMED-CT, RxNorm, drug interactions) | 5.x |
| **Cache** | Redis | Embedding cache, session cache, rate limiting | 7+ |
| **Task Queue** | Celery + Redis | Background tasks (batch T-GNN updates, retraining) | 5.4+ |
| **Data Validation** | Pydantic v2 | Schema enforcement across all data models | 2.x |
| **ORM** | SQLAlchemy 2.0 | Database interaction layer | 2.0+ |
| **Migrations** | Alembic | Database schema migrations | 1.13+ |
| **T-GNN** | PyTorch + PyTorch Geometric (PyG) | Temporal Graph Neural Network | 2.3+ / 2.5+ |
| **NLP / Medical** | spaCy + scispaCy + MedCAT | Medical entity extraction (NER) | - |
| **FHIR Client** | fhir.resources + fhirclient | HL7 FHIR data parsing | - |

### 2.2 Frontend Stack

| Layer | Technology | Purpose | Phase |
|-------|-----------|---------|-------|
| **MVP UI** | Streamlit | Rapid prototyping, internal testing | Phase 0–2 |
| **Production UI** | Next.js 14 + React 18 + TypeScript | Production clinical dashboard | Phase 3+ |
| **Styling** | Tailwind CSS + shadcn/ui | Component library + responsive design | Phase 3+ |
| **Charts** | Recharts + D3.js | Patient timeline, confidence plots | Phase 3+ |
| **Graph Visualization** | Cytoscape.js | Temporal graph + knowledge graph visualization | Phase 4+ |
| **State Management** | Zustand | Lightweight frontend state | Phase 3+ |
| **API Client** | TanStack Query (React Query) | Server state management + caching | Phase 3+ |
| **Real-time** | WebSockets (FastAPI) | Live agent reasoning stream | Phase 3+ |
| **Auth** | NextAuth.js + OAuth2/OIDC | Clinician authentication | Phase 3+ |

### 2.3 Infrastructure & DevOps

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Containerization** | Docker + Docker Compose | Local development + packaging |
| **Orchestration** | Kubernetes (GKE) | Production deployment |
| **CI/CD** | GitHub Actions | Automated testing + deployment |
| **Monitoring** | Prometheus + Grafana | System metrics, latency tracking |
| **Logging** | Structlog + ELK Stack | Structured logging + search |
| **Model Tracking** | MLflow | Experiment tracking, model versioning |
| **Data Versioning** | DVC | Dataset versioning linked to Git |
| **Secrets** | Google Secret Manager / HashiCorp Vault | API keys, credentials |
| **Load Testing** | Locust | Performance benchmarking |

---

## 3. Repository Structure

```
ns-tea/
│
├── README.md
├── pyproject.toml                    # Project config (dependencies, build)
├── Makefile                          # Common commands (run, test, lint, migrate)
├── docker-compose.yml                # Local dev environment
├── .env.example                      # Environment variable template
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint + test on PR
│       ├── cd.yml                    # Deploy on merge to main
│       └── eval.yml                  # Scheduled evaluation runs
│
├── src/
│   └── nstea/
│       ├── __init__.py
│       ├── config.py                 # Centralized configuration (Pydantic Settings)
│       ├── main.py                   # FastAPI application entry point
│       │
│       ├── api/                      # ═══ REST API Layer ═══
│       │   ├── __init__.py
│       │   ├── deps.py               # Dependency injection (DB sessions, services)
│       │   ├── middleware.py          # Auth, CORS, rate limiting, request logging
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── patients.py       # CRUD patient data
│       │       ├── analysis.py       # Trigger clinical analysis
│       │       ├── feedback.py       # Clinician feedback submission
│       │       ├── admin.py          # System health, config
│       │       └── websocket.py      # Real-time reasoning stream
│       │
│       ├── models/                   # ═══ Data Models (Pydantic + SQLAlchemy) ═══
│       │   ├── __init__.py
│       │   ├── patient.py            # Patient schema (input/output/DB)
│       │   ├── clinical.py           # Conditions, medications, labs
│       │   ├── analysis.py           # Analysis request/response
│       │   ├── recommendation.py     # Clinical recommendation output
│       │   ├── feedback.py           # Clinician feedback
│       │   └── db/
│       │       ├── __init__.py
│       │       ├── base.py           # SQLAlchemy base
│       │       ├── patient.py        # Patient ORM model
│       │       ├── analysis_log.py   # Audit log ORM model
│       │       └── feedback.py       # Feedback ORM model
│       │
│       ├── data_layer/               # ═══ Phase 1: Data Ingestion ═══
│       │   ├── __init__.py
│       │   ├── fhir_client.py        # FHIR resource parser
│       │   ├── preprocessor.py       # Standardize → ICD-10, RxNorm, LOINC
│       │   ├── entity_extractor.py   # scispaCy medical NER
│       │   └── schema_mapper.py      # Raw data → internal schema mapping
│       │
│       ├── retrieval/                # ═══ Phase 1: RAG Layer ═══
│       │   ├── __init__.py
│       │   ├── embedder.py           # Text → vector embedding
│       │   ├── vector_store.py       # Qdrant client wrapper
│       │   ├── retriever.py          # Query → Top-K documents
│       │   ├── context_builder.py    # Assemble grounded context
│       │   └── document_loader.py    # Load + chunk clinical guidelines
│       │
│       ├── agents/                   # ═══ Agno Agent Definitions ═══
│       │   ├── __init__.py
│       │   ├── orchestrator.py       # Root Workflow (main pipeline)
│       │   ├── triage_agent.py       # Complexity assessment + routing
│       │   ├── reasoning_agent.py    # Agent: clinical reasoning
│       │   ├── retrieval_agent.py    # Custom BaseAgent: RAG tool invocation
│       │   ├── safety_agent.py       # Custom BaseAgent: constraint validation
│       │   ├── confidence_agent.py   # Custom BaseAgent: confidence gating
│       │   ├── temporal_agent.py     # Custom BaseAgent: T-GNN embedding (Phase 4)
│       │   └── prompts/
│       │       ├── __init__.py
│       │       ├── system_prompts.py # Versioned system prompts
│       │       ├── reasoning.py      # Structured reasoning templates
│       │       └── safety.py         # Safety-specific prompts
│       │
│       ├── tools/                    # ═══ Agno Custom Tools ═══
│       │   ├── __init__.py
│       │   ├── drug_interaction.py   # Check drug-drug interactions
│       │   ├── lab_calculator.py     # Clinical calculators (eGFR, CHA₂DS₂-VASc)
│       │   ├── guideline_lookup.py   # RAG retrieval as Agno tool function
│       │   ├── contraindication.py   # Allergy + contraindication checker
│       │   └── icd_lookup.py         # ICD-10 code lookup
│       │
│       ├── safety/                   # ═══ Phase 1: Rule Engine / Phase 5: Full ═══
│       │   ├── __init__.py
│       │   ├── rule_engine.py        # Hard-coded clinical rules (Phase 1)
│       │   ├── constraint_engine.py  # Neo4j-backed symbolic engine (Phase 5)
│       │   ├── knowledge_graph.py    # Neo4j client + SNOMED/RxNorm queries
│       │   └── validator.py          # Unified validation interface
│       │
│       ├── temporal/                 # ═══ Phase 4: T-GNN ═══
│       │   ├── __init__.py
│       │   ├── graph_builder.py      # Patient history → NetworkX graph
│       │   ├── temporal_encoder.py   # Time-decay attention weights
│       │   ├── tgnn_model.py         # PyG Temporal GNN model
│       │   ├── embedding_cache.py    # Redis-backed embedding store
│       │   └── batch_updater.py      # Scheduled batch graph updates
│       │
│       ├── services/                 # ═══ Business Logic Layer ═══
│       │   ├── __init__.py
│       │   ├── analysis_service.py   # Orchestrate full analysis pipeline
│       │   ├── patient_service.py    # Patient data management
│       │   ├── feedback_service.py   # Process clinician feedback
│       │   └── audit_service.py      # Audit logging
│       │
│       └── core/                     # ═══ Shared Utilities ═══
│           ├── __init__.py
│           ├── database.py           # PostgreSQL connection + session factory
│           ├── redis_client.py       # Redis connection
│           ├── neo4j_client.py       # Neo4j connection
│           ├── exceptions.py         # Custom exception hierarchy
│           ├── logging.py            # Structured logging setup
│           └── metrics.py            # Prometheus metric definitions
│
├── frontend/                         # ═══ Frontend Application ═══
│   ├── streamlit_app/                # Phase 0–2: Streamlit MVP
│   │   ├── app.py                    # Main Streamlit entry
│   │   ├── pages/
│   │   │   ├── patient_input.py      # Patient data entry
│   │   │   ├── analysis.py           # Trigger + view analysis
│   │   │   └── feedback.py           # Submit feedback
│   │   └── components/
│   │       ├── patient_card.py       # Patient summary widget
│   │       └── recommendation.py     # Recommendation display widget
│   │
│   └── web/                          # Phase 3+: Next.js Production UI
│       ├── package.json
│       ├── next.config.js
│       ├── tailwind.config.ts
│       ├── tsconfig.json
│       ├── src/
│       │   ├── app/                  # Next.js App Router
│       │   │   ├── layout.tsx
│       │   │   ├── page.tsx          # Dashboard home
│       │   │   ├── patients/
│       │   │   │   ├── page.tsx      # Patient list
│       │   │   │   └── [id]/
│       │   │   │       ├── page.tsx  # Patient detail + timeline
│       │   │   │       └── analysis/
│       │   │   │           └── page.tsx  # Analysis view
│       │   │   ├── analysis/
│       │   │   │   └── [id]/
│       │   │   │       └── page.tsx  # Full analysis detail view
│       │   │   └── admin/
│       │   │       └── page.tsx      # System health + config
│       │   ├── components/
│       │   │   ├── ui/               # shadcn/ui components
│       │   │   ├── PatientCard.tsx
│       │   │   ├── PatientTimeline.tsx
│       │   │   ├── RecommendationPanel.tsx
│       │   │   ├── EvidencePanel.tsx
│       │   │   ├── ConfidenceMeter.tsx
│       │   │   ├── ReasoningTrace.tsx
│       │   │   ├── SafetyFlags.tsx
│       │   │   ├── FeedbackForm.tsx
│       │   │   └── TemporalGraph.tsx  # Phase 4+
│       │   ├── lib/
│       │   │   ├── api.ts            # API client
│       │   │   └── types.ts          # TypeScript types mirroring backend
│       │   └── hooks/
│       │       ├── useAnalysis.ts    # Analysis query hook
│       │       └── useWebSocket.ts   # Real-time reasoning stream
│       └── public/
│           └── icons/
│
├── data/                             # ═══ Data Assets ═══
│   ├── guidelines/                   # Clinical guideline documents (PDF/text)
│   ├── ontologies/                   # SNOMED-CT, RxNorm, ICD-10 mappings
│   ├── rules/                        # Rule engine YAML definitions
│   │   ├── drug_interactions.yml
│   │   ├── contraindications.yml
│   │   └── dosage_limits.yml
│   └── synthetic/                    # Synthetic test patient data
│       ├── patients.json
│       └── cases.json
│
├── tests/                            # ═══ Test Suite ═══
│   ├── conftest.py                   # Shared fixtures
│   ├── unit/
│   │   ├── test_preprocessor.py
│   │   ├── test_retriever.py
│   │   ├── test_rule_engine.py
│   │   ├── test_constraint_engine.py
│   │   └── test_graph_builder.py
│   ├── integration/
│   │   ├── test_analysis_pipeline.py
│   │   ├── test_agent_orchestration.py
│   │   └── test_api_endpoints.py
│   ├── safety/                       # Dedicated safety tests
│   │   ├── test_hallucination.py     # Adversarial hallucination checks
│   │   ├── test_contraindications.py # Known dangerous combinations
│   │   └── test_bias.py             # Demographic bias detection
│   └── evaluation/
│       ├── eval_medqa.py             # MedQA benchmark
│       ├── eval_safety.py            # Safety violation rate
│       └── eval_latency.py           # Performance benchmarks
│
├── notebooks/                        # ═══ Research & Exploration ═══
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_tgnn_prototype.ipynb
│   └── 04_evaluation_results.ipynb
│
├── scripts/                          # ═══ Utility Scripts ═══
│   ├── seed_knowledge_graph.py       # Populate Neo4j with SNOMED/RxNorm
│   ├── index_guidelines.py           # Chunk + embed + index guidelines
│   ├── generate_synthetic_data.py    # Create test patients
│   └── run_evaluation.py             # Run full evaluation suite
│
├── mlops/                            # ═══ MLOps Configuration ═══
│   ├── mlflow/
│   │   └── mlflow_config.yml
│   ├── dvc/
│   │   └── dvc.yaml
│   └── training/
│       └── train_tgnn.py             # T-GNN training script
│
└── infra/                            # ═══ Infrastructure as Code ═══
    ├── docker/
    │   ├── Dockerfile.backend
    │   ├── Dockerfile.frontend
    │   └── Dockerfile.tgnn
    ├── k8s/
    │   ├── namespace.yml
    │   ├── backend-deployment.yml
    │   ├── frontend-deployment.yml
    │   ├── redis-deployment.yml
    │   └── ingress.yml
    └── terraform/
        ├── main.tf
        ├── gke.tf
        └── variables.tf
```

---

## 4. Data Models & Schemas

### 4.1 Core Patient Model

```python
# src/nstea/models/patient.py

class PatientInput(BaseModel):
    """What the clinician submits."""
    patient_id: str
    age: int
    sex: Literal["male", "female", "other"]
    conditions: list[Condition]           # Active diagnoses
    medications: list[Medication]          # Current medications
    allergies: list[Allergy]              # Known allergies
    symptoms: list[Symptom]               # Presenting symptoms
    lab_results: list[LabResult]          # Recent labs
    vitals: Optional[Vitals] = None       # Current vitals
    history: list[ClinicalEvent]          # Chronological history
    clinician_query: str                  # Natural language question

class Condition(BaseModel):
    name: str
    icd10_code: Optional[str] = None
    onset_date: Optional[date] = None
    status: Literal["active", "resolved", "chronic"]

class Medication(BaseModel):
    name: str
    rxnorm_code: Optional[str] = None
    dosage: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

class Allergy(BaseModel):
    substance: str
    reaction: Optional[str] = None
    severity: Literal["mild", "moderate", "severe"]

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

class ClinicalEvent(BaseModel):
    event_type: Literal["diagnosis", "medication", "procedure", "lab", "visit", "imaging"]
    description: str
    date: date
    details: Optional[dict] = None

class Vitals(BaseModel):
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[int] = None
    spo2: Optional[float] = None
```

### 4.2 Analysis Response Model

```python
# src/nstea/models/recommendation.py

class AnalysisResponse(BaseModel):
    """What the clinician sees."""
    analysis_id: str                       # Unique analysis identifier
    patient_id: str
    timestamp: datetime

    # Primary Output
    diagnosis: DiagnosisOutput
    recommendations: list[Recommendation]
    
    # Explainability
    reasoning_trace: ReasoningTrace
    evidence: list[Evidence]
    
    # Safety
    safety_flags: list[SafetyFlag]
    constraint_violations: list[ConstraintViolation]
    
    # Confidence
    confidence: ConfidenceScore
    requires_human_review: bool
    escalation_reason: Optional[str] = None

class DiagnosisOutput(BaseModel):
    primary: str                           # Most likely diagnosis
    differential: list[DifferentialDx]     # Ranked alternatives
    
class DifferentialDx(BaseModel):
    diagnosis: str
    probability: float                     # 0.0 – 1.0
    supporting_evidence: list[str]
    contradicting_evidence: list[str]

class Recommendation(BaseModel):
    action: str                            # e.g., "Start aspirin 81mg daily"
    category: Literal["medication", "test", "procedure", "referral", "monitoring"]
    urgency: Literal["stat", "urgent", "routine"]
    rationale: str                         # Why this is recommended
    guideline_source: Optional[str] = None # Which guideline supports this

class ReasoningTrace(BaseModel):
    steps: list[ReasoningStep]
    
class ReasoningStep(BaseModel):
    step_number: int
    description: str                       # What the agent did
    input_summary: str                     # What it considered
    output_summary: str                    # What it concluded
    agent_name: str                        # Which Agno agent performed this

class Evidence(BaseModel):
    source: str                            # Guideline name / paper
    snippet: str                           # Relevant text excerpt
    relevance_score: float                 # How relevant to this case
    url: Optional[str] = None

class SafetyFlag(BaseModel):
    level: Literal["info", "warning", "critical"]
    message: str
    related_recommendation: Optional[str] = None

class ConstraintViolation(BaseModel):
    rule: str                              # What rule was violated
    description: str                       # Human-readable explanation
    action_blocked: str                    # What was prevented

class ConfidenceScore(BaseModel):
    overall: float                         # 0.0 – 1.0
    evidence_strength: float               # How well RAG supported the conclusion
    model_certainty: float                 # LLM's self-assessed confidence
    guideline_compliance: float            # How well recommendation matches guidelines
```

### 4.3 Feedback Model

```python
# src/nstea/models/feedback.py

class ClinicalFeedback(BaseModel):
    """Clinician's response to a recommendation."""
    analysis_id: str
    clinician_id: str
    decision: Literal["accept", "modify", "reject"]
    modifications: Optional[str] = None    # What was changed
    rejection_reason: Optional[str] = None
    correct_diagnosis: Optional[str] = None
    notes: Optional[str] = None
    timestamp: datetime
```

---

## 5. Backend Architecture (Detailed)

### 5.1 Service Layer Pattern

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Router Layer                       │
│  /api/v1/patients  |  /api/v1/analyze  |  /api/v1/feedback       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                        Service Layer                              │
│  AnalysisService  |  PatientService  |  FeedbackService          │
│  (business logic, validation, orchestration)                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│              Agno Agent Orchestration Layer                         │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Workflow (Root Pipeline)                                 │      │
│  │    ├── TriageAgent (complexity assessment)               │      │
│  │    ├── Team (data enrichment)                            │      │
│  │    │     ├── RetrievalAgent (RAG)                        │      │
│  │    │     └── TemporalAgent (T-GNN) [Phase 4]            │      │
│  │    ├── ReasoningAgent (LLM clinical reasoning)           │      │
│  │    ├── SafetyAgent (constraint validation)               │      │
│  │    └── ConfidenceAgent (gating + output formatting)      │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                    │
│  Workflow (loop) wraps [ReasoningAgent → SafetyAgent] for retry   │
│  constraint violations (max 3 iterations)                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    Data Access Layer                               │
│  PostgreSQL  |  Qdrant  |  Neo4j  |  Redis                       │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Agno Agent Design (Core Architecture)

The NS-TEA pipeline maps directly to Agno agent primitives:

```python
# Conceptual architecture showing Agno agent composition

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.huggingface import HuggingFace
# from agno.team import Team  # For parallel execution
# from agno.workflow import Workflow  # For sequential/loop orchestration

# ═══ CUSTOM AGENTS (BaseAgent extensions) ═══

class TriageAgent(BaseAgent):
    """Assesses complexity, determines pipeline path."""
    # Simple case → skip T-GNN, use RAG + LLM + rules
    # Complex case → full pipeline with T-GNN + multi-retrieval

class RetrievalAgent(BaseAgent):
    """Executes RAG: query → embed → vector search → context assembly."""
    # Uses Qdrant vector store
    # Outputs grounded context to session state

class TemporalAgent(BaseAgent):
    """Fetches/generates T-GNN embedding for patient."""
    # Phase 4: fetches cached embedding or triggers batch compute
    # Outputs temporal embedding to session state

class SafetyAgent(BaseAgent):
    """Validates proposed actions against constraint engine."""
    # Phase 1: YAML rule engine
    # Phase 5: Neo4j symbolic engine
    # Escalates (returns Event with escalate=True) if violation found

class ConfidenceAgent(BaseAgent):
    """Calculates confidence score + determines HITL escalation."""
    # Aggregates: evidence_strength + model_certainty + guideline_compliance
    # If confidence < threshold → sets requires_human_review = True

# ═══ LLM AGENTS ═══

reasoning_agent = Agent(
    name="ClinicalReasoner",
    model=Ollama(id="gpt-oss:20b"),  # or HuggingFace(id="...")
    instructions="...",  # Structured reasoning prompt
    tools=[guideline_lookup_tool, drug_interaction_tool, lab_calculator_tool],
    output_key="candidate_recommendation"
)

# ═══ WORKFLOW ORCHESTRATION ═══

# Safety loop: if constraint engine rejects, LLM retries
safety_loop = Workflow(  # loop pattern
    name="SafetyValidationLoop",
    max_iterations=3,
    sub_agents=[reasoning_agent, safety_agent, check_safety_status]
)

# Data enrichment in parallel
data_enrichment = Team(
    name="DataEnrichment",
    sub_agents=[retrieval_agent, temporal_agent]
)

# Main pipeline
root_agent = Workflow(  # sequential
    name="NS_TEA_Pipeline",
    sub_agents=[
        triage_agent,           # Step 1: Assess complexity
        data_enrichment,        # Step 2: RAG + T-GNN in parallel
        safety_loop,            # Step 3: Reason → Validate (with retry)
        confidence_agent        # Step 4: Score + format output
    ]
)
```

### 5.3 Agno Session State Flow

The pipeline communicates via Agno's shared session state:

```
State Key                          Written By            Read By
─────────────────────────────────────────────────────────────────
patient_data                       API Layer             TriageAgent
complexity_level                   TriageAgent           DataEnrichment
retrieved_context                  RetrievalAgent        ReasoningAgent
temporal_embedding                 TemporalAgent         ReasoningAgent
candidate_recommendation           ReasoningAgent        SafetyAgent
safety_result                      SafetyAgent           CheckStatus
safety_violations                  SafetyAgent           ConfidenceAgent
validated_recommendation           SafetyAgent           ConfidenceAgent
confidence_score                   ConfidenceAgent       API Layer
requires_human_review              ConfidenceAgent       API Layer
final_output                       ConfidenceAgent       API Layer
reasoning_trace                    All agents            API Layer
```

### 5.4 Database Schema (PostgreSQL)

```sql
-- Core tables

CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    demographics JSONB NOT NULL,           -- age, sex, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE clinical_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id),
    data_type VARCHAR(50) NOT NULL,        -- condition, medication, lab, etc.
    data JSONB NOT NULL,                   -- Flexible clinical data
    event_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE analysis_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID REFERENCES patients(id),
    clinician_id VARCHAR(255) NOT NULL,
    request JSONB NOT NULL,                -- Full input
    response JSONB NOT NULL,               -- Full output
    reasoning_trace JSONB,                 -- Step-by-step trace
    safety_flags JSONB,                    -- Any safety issues
    confidence_score FLOAT,
    latency_ms INTEGER,                    -- Total processing time
    model_version VARCHAR(100),            -- Which model was used
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analysis_logs(id),
    clinician_id VARCHAR(255) NOT NULL,
    decision VARCHAR(20) NOT NULL,         -- accept/modify/reject
    modifications TEXT,
    rejection_reason TEXT,
    correct_diagnosis TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_clinical_data_patient ON clinical_data(patient_id);
CREATE INDEX idx_clinical_data_type ON clinical_data(data_type);
CREATE INDEX idx_analysis_logs_patient ON analysis_logs(patient_id);
CREATE INDEX idx_analysis_logs_created ON analysis_logs(created_at);
CREATE INDEX idx_feedback_analysis ON feedback(analysis_id);
```

---

## 6. Frontend UI/UX Design

### 6.1 Design Philosophy

| Principle | Rationale |
|-----------|-----------|
| **Clinical-First** | UI designed for clinical workflow, not tech demos |
| **Information Density** | Clinicians need dense, scannable information |
| **Clear Hierarchy** | Diagnosis → Evidence → Safety → Action |
| **Mandatory Friction** | Deliberate friction before accepting high-risk recommendations |
| **Accessibility** | WCAG 2.1 AA compliant; works on tablets in hospital settings |

### 6.2 Page Architecture & Wireframes

#### Screen 1: Dashboard (Landing Page)

```
┌─────────────────────────────────────────────────────────────────┐
│  NS-TEA Clinical Decision Support          🔔 Alerts    👤 Dr. X │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─── Quick Analysis ────────────────────────────────────────┐   │
│  │  Search Patient: [________________] or [+ New Analysis]    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Recent Analyses ───────────────────────────────────────┐   │
│  │                                                            │   │
│  │  Patient        Diagnosis           Confidence   Status    │   │
│  │  ──────────────────────────────────────────────────────    │   │
│  │  John D, 65     MI (suspected)      ●●●●○ 92%   ✅ Accepted│   │
│  │  Sarah M, 42    UTI                 ●●●○○ 78%   ⏳ Pending │   │
│  │  Alex T, 71     COPD Exacerbation   ●●○○○ 53%   ⚠️ Review │   │
│  │                                                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── System Health ─────────┐  ┌─── Stats ─────────────────┐   │
│  │  API: ● Online             │  │  Today: 23 analyses        │   │
│  │  Avg Latency: 3.2s         │  │  Accepted: 87%             │   │
│  │  Safety Blocks: 2 today    │  │  Modified: 9%              │   │
│  └────────────────────────────┘  │  Rejected: 4%              │   │
│                                  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Screen 2: Patient Analysis Input

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back    New Clinical Analysis                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─── Patient Information ───────────────────────────────────┐   │
│  │                                                            │   │
│  │  Patient ID: [________]  Age: [__]  Sex: [▼ Male     ]    │   │
│  │                                                            │   │
│  │  Active Conditions:              Current Medications:      │   │
│  │  [+ Add Condition]               [+ Add Medication]        │   │
│  │  • Diabetes Type 2 (E11.9)       • Metformin 500mg BID    │   │
│  │  • Hypertension (I10)            • Lisinopril 10mg QD     │   │
│  │                                                            │   │
│  │  Allergies:                      Presenting Symptoms:      │   │
│  │  [+ Add Allergy]                 [+ Add Symptom]           │   │
│  │  • Aspirin (severe: anaphylaxis) • Chest pain (acute)      │   │
│  │                                  • Shortness of breath     │   │
│  │                                                            │   │
│  │  Recent Labs:                                              │   │
│  │  [+ Add Lab Result]                                        │   │
│  │  • Troponin I: 0.8 ng/mL ⚠️ HIGH  (2026-04-07)           │   │
│  │  • HbA1c: 7.2%                     (2026-03-15)           │   │
│  │                                                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Clinical History Timeline ─────────────────────────────┐   │
│  │  [+ Add Event]                                             │   │
│  │                                                            │   │
│  │  2025-10 ──●── Started NSAIDs for joint pain               │   │
│  │  2025-06 ──●── Diagnosed Type 2 Diabetes                   │   │
│  │  2024-11 ──●── Hypertension diagnosed                      │   │
│  │  2024-03 ──●── Annual physical (normal)                    │   │
│  │                                                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Your Question ─────────────────────────────────────────┐   │
│  │  What is the likely diagnosis and recommended treatment     │   │
│  │  plan for this patient?                                     │   │
│  │  [___________________________________________________]     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│                              [ 🔍 Analyze Patient ]               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Screen 3: Analysis Results (Core Screen)

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back    Analysis Results    Patient: John D, 65    ID: A-1847 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─── ⚠️ SAFETY ALERTS ─────────────────────────────────────┐   │
│  │  🔴 CRITICAL: Aspirin contraindicated — patient has known  │   │
│  │     aspirin allergy (anaphylaxis). Alternative: Clopidogrel │   │
│  │  🟡 WARNING: NSAIDs (started 6mo ago) ↑ cardiovascular risk│   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Diagnosis ──────────────────┐  ┌─── Confidence ────────┐  │
│  │                                 │  │                        │  │
│  │  PRIMARY: Myocardial Infarction │  │  Overall: ████████░░   │  │
│  │  (STEMI suspected)              │  │           85%          │  │
│  │                                 │  │                        │  │
│  │  Differential:                  │  │  Evidence:  ●●●●○ 90%  │  │
│  │  1. NSTEMI         (0.72)       │  │  Model:     ●●●○○ 78%  │  │
│  │  2. Unstable Angina (0.15)      │  │  Guideline: ●●●●○ 88%  │  │
│  │  3. GERD            (0.08)      │  │                        │  │
│  │  4. Anxiety          (0.05)     │  │  ⚠️ Human Review       │  │
│  │                                 │  │     Recommended        │  │
│  └─────────────────────────────────┘  └────────────────────────┘  │
│                                                                   │
│  ┌─── Recommendations ──────────────────────────────────────┐    │
│  │                                                            │   │
│  │  🟢 STAT Actions:                                         │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Start Clopidogrel 300mg loading dose               │ │   │
│  │  │    Rationale: Antiplatelet therapy indicated for ACS;  │ │   │
│  │  │    aspirin contraindicated due to allergy              │ │   │
│  │  │    Guideline: ACC/AHA 2023 ACS Guidelines §4.2        │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ 2. Order 12-lead ECG immediately                      │ │   │
│  │  │    Rationale: Confirm STEMI vs NSTEMI; troponin       │ │   │
│  │  │    elevated at 0.8 ng/mL (normal <0.04)               │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ 3. Admit to CCU / Cardiac monitoring                  │ │   │
│  │  │    Rationale: High-risk presentation requiring        │ │   │
│  │  │    continuous telemetry                                │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  │                                                            │   │
│  │  🟡 URGENT Actions:                                       │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ 4. Discontinue NSAIDs — elevated CV risk              │ │   │
│  │  │ 5. Cardiology consult                                 │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Reasoning Trace (Expandable) ─────────────────────────┐   │
│  │  ▶ Step 1: Triage — Classified as HIGH complexity          │   │
│  │  ▶ Step 2: Retrieved 8 relevant guidelines (ACC/AHA...)    │   │
│  │  ▶ Step 3: Temporal insight — NSAIDs started 6 months ago  │   │
│  │            contributes to elevated cardiovascular risk       │   │
│  │  ▶ Step 4: Hypothesis generation — MI (0.72), UA (0.15)... │   │
│  │  ▶ Step 5: Safety check — BLOCKED aspirin (allergy)        │   │
│  │            → Substituted with Clopidogrel                   │   │
│  │  ▶ Step 6: Confidence scoring — 85% overall                │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Supporting Evidence ───────────────────────────────────┐   │
│  │  📄 ACC/AHA 2023 ACS Guidelines §4.2 — "Dual antiplatelet │   │
│  │     therapy is recommended..." (relevance: 0.94)           │   │
│  │  📄 ESC 2023 NSTEMI Management — "Troponin >0.04 ng/mL..."│   │
│  │     (relevance: 0.91)                                       │   │
│  │  📄 FDA NSAID CV Risk Warning — "NSAIDs increase risk..."  │   │
│  │     (relevance: 0.87)                                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Temporal Graph (Phase 4+) ─────────────────────────────┐   │
│  │  [Interactive Cytoscape.js visualization]                   │   │
│  │                                                            │   │
│  │  (2024-03)──(2024-11)──(2025-06)──(2025-10)──(2026-04)   │   │
│  │   Physical   HTN dx    DM2 dx    NSAIDs     Chest pain    │   │
│  │                ╰─────────╯  ╰────────╯  ╰────────╯        │   │
│  │              causal link   temporal    CV risk path         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Your Decision ─────────────────────────────────────────┐   │
│  │                                                            │   │
│  │  [ ✅ Accept ]   [ ✏️ Modify ]   [ ❌ Reject ]            │   │
│  │                                                            │   │
│  │  (If Modify/Reject):                                       │   │
│  │  Notes: [_______________________________________________]  │   │
│  │  Correct Diagnosis (if different): [___________________]   │   │
│  │                                                            │   │
│  │                    [ Submit Feedback ]                      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Screen 4: Admin / System Health

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back    System Administration                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─── System Status ─────────────────────────────────────────┐   │
│  │  Backend API:      ● Online   (latency: 3.2s avg)         │   │
│  │  Vector DB:        ● Online   (12,847 documents indexed)  │   │
│  │  Knowledge Graph:  ● Online   (342K nodes, 1.2M edges)    │   │
│  │  T-GNN Cache:      ● Online   (1,204 embeddings cached)   │   │
│  │  Redis:            ● Online   (memory: 2.1GB / 8GB)       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Performance Metrics (Last 30 days) ────────────────────┐   │
│  │  [Latency histogram chart]                                 │   │
│  │  P50: 2.8s  |  P95: 8.1s  |  P99: 14.2s                  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Safety Metrics ────────────────────────────────────────┐   │
│  │  Constraint violations caught: 147 (6.2% of analyses)      │   │
│  │  Human escalations: 89 (3.8%)                              │   │
│  │  Clinician agreement rate: 87%                             │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─── Feedback Summary ──────────────────────────────────────┐   │
│  │  [Pie chart: Accept 87% | Modify 9% | Reject 4%]          │   │
│  │  Common rejection reasons: [bar chart]                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Frontend Phasing

| Phase | UI Technology | What Gets Built |
|-------|--------------|-----------------|
| **Phase 0** | Streamlit | Single page: text input → raw JSON output |
| **Phase 1** | Streamlit | Patient form + formatted recommendation display |
| **Phase 2** | Streamlit | + History view, feedback form, basic metrics |
| **Phase 3** | Next.js | Full production UI with all 4 screens above |
| **Phase 4** | Next.js | + Temporal graph visualization (Cytoscape.js) |
| **Phase 5** | Next.js | + Knowledge graph explorer, constraint trace view |
| **Phase 6** | Next.js | + Admin dashboard, audit logs, drift alerts |

---

## 7. Agno Agent Architecture

### 7.1 Agent Hierarchy (Visual)

```
Root: Workflow ("NS_TEA_Pipeline")
│
├── 1. TriageAgent (Custom BaseAgent)
│       ├── Reads: patient_data
│       ├── Writes: complexity_level, pipeline_config
│       └── Logic: rule-based complexity scoring (symptom count,
│                  comorbidity count, lab abnormalities)
│
├── 2. Team ("DataEnrichment")
│       ├── RetrievalAgent (Custom Agent)
│       │       ├── Reads: patient_data, complexity_level
│       │       ├── Writes: retrieved_context
│       │       └── Tools: guideline_lookup (tool function)
│       │
│       └── TemporalAgent (Custom BaseAgent) [Phase 4+]
│               ├── Reads: patient_data.history
│               ├── Writes: temporal_embedding, temporal_insights
│               └── Logic: graph_builder → embedding_cache lookup
│
├── 3. Workflow ("SafetyValidationLoop", loop pattern, max_iterations=3)
│       │
│       ├── ReasoningAgent (Agent)
│       │       ├── Model: gpt-oss:20b (Ollama) / HuggingFace
│       │       ├── Reads: patient_data, retrieved_context,
│       │       │          temporal_embedding, safety_feedback
│       │       ├── Writes: candidate_recommendation, reasoning_trace
│       │       └── Tools:
│       │             ├── drug_interaction_checker (tool function)
│       │             ├── lab_calculator (tool function)
│       │             ├── guideline_lookup (tool function)
│       │             └── icd_lookup (tool function)
│       │
│       ├── SafetyAgent (Custom BaseAgent)
│       │       ├── Reads: candidate_recommendation, patient_data
│       │       ├── Writes: safety_result, safety_violations, safety_feedback
│       │       └── Logic:
│       │             Phase 1: YAML rule engine
│       │             Phase 5: Neo4j constraint engine
│       │
│       └── CheckSafetyStatus (Custom BaseAgent)
│               ├── Reads: safety_result
│               └── Logic: if safe → escalate (exit loop)
│                          if unsafe → continue loop (retry)
│
└── 4. ConfidenceAgent (Custom BaseAgent)
        ├── Reads: candidate_recommendation, safety_result,
        │          retrieved_context, reasoning_trace
        ├── Writes: final_output, confidence_score, requires_human_review
        └── Logic: weighted scoring → threshold gating
```

### 7.2 Agno Tools (Tool Function Definitions)

```python
# Each tool is a Python function decorated as an Agno tool

# Tool 1: Drug Interaction Checker
async def check_drug_interactions(
    drug_name: str,
    patient_medications: list[str]
) -> dict:
    """Check if a proposed drug has interactions with patient's current medications.
    Returns interaction severity and description."""

# Tool 2: Lab Calculator
async def calculate_clinical_score(
    calculator_name: str,  # "eGFR", "CHA2DS2-VASc", "MELD", "Wells"
    parameters: dict
) -> dict:
    """Calculate standard clinical scores from lab values and patient data."""

# Tool 3: Guideline Lookup
async def lookup_guideline(
    query: str,
    specialty: Optional[str] = None,
    max_results: int = 5
) -> list[dict]:
    """Search clinical guidelines via RAG. Returns relevant excerpts
    with source attribution and relevance scores."""

# Tool 4: ICD-10 Lookup
async def lookup_icd10(
    condition_description: str
) -> dict:
    """Map a condition description to its ICD-10 code and metadata."""

# Tool 5: Contraindication Checker
async def check_contraindications(
    proposed_action: str,
    patient_allergies: list[str],
    patient_conditions: list[str]
) -> dict:
    """Check if a proposed treatment has contraindications given
    patient allergies and conditions."""
```

### 7.3 Agno Session & State Management

```python
# Session configuration for NS-TEA

from agno.memory import AgnoMemory  # Dev (in-memory session)
# from agno.memory import PostgresMemory  # Production

# State prefix conventions:
# - No prefix:     Analysis-scoped (one patient analysis)
# - "user:"        Clinician preferences
# - "app:"         Global system config
# - "temp:"        Within-invocation intermediates

STATE_SCHEMA = {
    # Input (set by API layer before agent invocation)
    "patient_data": "...",              # Full PatientInput JSON
    "clinician_query": "...",           # Natural language question
    
    # Triage output
    "temp:complexity_level": "...",     # simple | moderate | complex
    "temp:pipeline_config": "...",      # Which agents to activate
    
    # Retrieval output
    "temp:retrieved_context": "...",    # List of retrieved documents
    "temp:retrieval_scores": "...",     # Relevance scores
    
    # Temporal output (Phase 4+)
    "temp:temporal_embedding": "...",   # Dense vector from T-GNN
    "temp:temporal_insights": "...",    # Human-readable temporal findings
    
    # Reasoning output
    "temp:candidate_recommendation": "...",
    "temp:reasoning_trace": "...",
    
    # Safety output
    "temp:safety_result": "...",        # pass | fail
    "temp:safety_violations": "...",    # List of violations
    "temp:safety_feedback": "...",      # Feedback for LLM retry
    
    # Final output
    "temp:final_output": "...",         # AnalysisResponse JSON
    "temp:confidence_score": "...",
    "temp:requires_human_review": "...",
    
    # Persistent
    "user:preferences": "...",          # Clinician display preferences
    "app:confidence_threshold": 0.75,   # System-wide threshold
    "app:model_version": "...",
}
```

---

## 8. API Contracts

### 8.1 REST Endpoints

```
BASE URL: /api/v1

# ═══ Patient Management ═══
POST   /patients                    # Create/register patient
GET    /patients/{patient_id}       # Get patient data
PUT    /patients/{patient_id}       # Update patient data
GET    /patients                    # List patients (paginated)

# ═══ Clinical Analysis ═══
POST   /analyze                     # Submit analysis request
GET    /analyze/{analysis_id}       # Get analysis result
GET    /analyze/history             # List past analyses (paginated)

# ═══ Feedback ═══
POST   /feedback                    # Submit clinician feedback
GET    /feedback/{analysis_id}      # Get feedback for an analysis

# ═══ System ═══
GET    /health                      # Health check
GET    /health/detailed             # Detailed component health
GET    /metrics                     # Prometheus metrics endpoint

# ═══ WebSocket ═══
WS     /ws/analyze/{analysis_id}    # Real-time reasoning stream
```

### 8.2 Key Request/Response Examples

**POST /api/v1/analyze**

Request:
```json
{
  "patient_id": "P-12345",
  "age": 65,
  "sex": "male",
  "conditions": [
    {"name": "Type 2 Diabetes", "icd10_code": "E11.9", "status": "active"},
    {"name": "Hypertension", "icd10_code": "I10", "status": "active"}
  ],
  "medications": [
    {"name": "Metformin", "dosage": "500mg BID"},
    {"name": "Lisinopril", "dosage": "10mg QD"}
  ],
  "allergies": [
    {"substance": "Aspirin", "reaction": "anaphylaxis", "severity": "severe"}
  ],
  "symptoms": [
    {"description": "chest pain", "onset": "2 hours ago", "severity": "severe"},
    {"description": "shortness of breath", "severity": "moderate"}
  ],
  "lab_results": [
    {"test_name": "Troponin I", "value": 0.8, "unit": "ng/mL",
     "reference_range": "<0.04", "date": "2026-04-07", "is_abnormal": true}
  ],
  "history": [
    {"event_type": "medication", "description": "Started NSAIDs for joint pain",
     "date": "2025-10-15"},
    {"event_type": "diagnosis", "description": "Diagnosed Type 2 Diabetes",
     "date": "2025-06-20"}
  ],
  "clinician_query": "What is the likely diagnosis and recommended treatment?"
}
```

Response (202 Accepted → poll or WebSocket):
```json
{
  "analysis_id": "A-98765",
  "status": "processing",
  "ws_url": "/ws/analyze/A-98765"
}
```

**GET /api/v1/analyze/A-98765** (after completion):
```json
{
  "analysis_id": "A-98765",
  "patient_id": "P-12345",
  "timestamp": "2026-04-07T14:32:00Z",
  "latency_ms": 4200,
  "diagnosis": {
    "primary": "Myocardial Infarction (NSTEMI suspected)",
    "differential": [
      {"diagnosis": "NSTEMI", "probability": 0.72,
       "supporting_evidence": ["Elevated troponin (0.8 ng/mL)", "Chest pain", "CV risk factors"],
       "contradicting_evidence": []},
      {"diagnosis": "Unstable Angina", "probability": 0.15,
       "supporting_evidence": ["Chest pain", "CV risk factors"],
       "contradicting_evidence": ["Troponin elevated suggests myocardial injury"]},
      {"diagnosis": "GERD", "probability": 0.08,
       "supporting_evidence": ["Chest pain can mimic"],
       "contradicting_evidence": ["Troponin elevation", "Acuity of presentation"]},
      {"diagnosis": "Anxiety/Panic", "probability": 0.05,
       "supporting_evidence": ["SOB present"],
       "contradicting_evidence": ["Troponin elevation", "Age + risk factors"]}
    ]
  },
  "recommendations": [
    {"action": "Start Clopidogrel 300mg loading dose",
     "category": "medication", "urgency": "stat",
     "rationale": "Antiplatelet therapy indicated; aspirin contraindicated due to anaphylaxis allergy",
     "guideline_source": "ACC/AHA 2023 ACS Guidelines §4.2"},
    {"action": "Order 12-lead ECG immediately",
     "category": "test", "urgency": "stat",
     "rationale": "Differentiate STEMI vs NSTEMI; troponin 20x upper limit of normal"},
    {"action": "Admit to CCU with continuous cardiac monitoring",
     "category": "procedure", "urgency": "stat",
     "rationale": "High-risk ACS presentation requiring telemetry"},
    {"action": "Discontinue NSAIDs",
     "category": "medication", "urgency": "urgent",
     "rationale": "NSAIDs increase cardiovascular risk; FDA black box warning"},
    {"action": "Cardiology consult",
     "category": "referral", "urgency": "urgent",
     "rationale": "Evaluation for cardiac catheterization"}
  ],
  "reasoning_trace": {
    "steps": [
      {"step_number": 1, "agent_name": "TriageAgent",
       "description": "Complexity assessment",
       "input_summary": "5 conditions/meds, 2 acute symptoms, 1 critical lab",
       "output_summary": "COMPLEX — full pipeline activated"},
      {"step_number": 2, "agent_name": "RetrievalAgent",
       "description": "Retrieved 8 clinical guidelines",
       "input_summary": "Query: chest pain + elevated troponin + diabetes + HTN",
       "output_summary": "Top matches: ACC/AHA ACS (0.94), ESC NSTEMI (0.91), FDA NSAID Warning (0.87)"},
      {"step_number": 3, "agent_name": "TemporalAgent",
       "description": "Temporal analysis of patient history",
       "input_summary": "6 events spanning 2024-03 to 2026-04",
       "output_summary": "KEY INSIGHT: NSAID use (6mo) temporally linked to current cardiovascular event; elevated CV risk trajectory"},
      {"step_number": 4, "agent_name": "ClinicalReasoner",
       "description": "Hypothesis generation and treatment planning",
       "input_summary": "Patient context + 8 guidelines + temporal insights",
       "output_summary": "Primary: MI (0.72). Initially proposed Aspirin — sent to safety check"},
      {"step_number": 5, "agent_name": "SafetyAgent",
       "description": "Constraint validation — VIOLATION DETECTED",
       "input_summary": "Proposed: Aspirin. Patient allergy: Aspirin (anaphylaxis, severe)",
       "output_summary": "BLOCKED Aspirin. Feedback: Use Clopidogrel as alternative antiplatelet"},
      {"step_number": 6, "agent_name": "ClinicalReasoner",
       "description": "Revised recommendation (iteration 2)",
       "input_summary": "Safety feedback: substitute Aspirin → Clopidogrel",
       "output_summary": "Updated plan with Clopidogrel 300mg loading dose"},
      {"step_number": 7, "agent_name": "SafetyAgent",
       "description": "Re-validation — PASSED",
       "input_summary": "Revised plan: Clopidogrel + ECG + CCU admission",
       "output_summary": "No violations detected. All actions safe."},
      {"step_number": 8, "agent_name": "ConfidenceAgent",
       "description": "Confidence scoring",
       "input_summary": "Evidence: 0.90, Model: 0.78, Guideline: 0.88",
       "output_summary": "Overall: 0.85. Above threshold (0.75). Human review still recommended for STAT actions."}
    ]
  },
  "evidence": [
    {"source": "ACC/AHA 2023 ACS Guidelines §4.2", "relevance_score": 0.94,
     "snippet": "Dual antiplatelet therapy is recommended for patients with ACS. In patients with aspirin allergy, clopidogrel monotherapy is an acceptable alternative."},
    {"source": "ESC 2023 NSTEMI Management", "relevance_score": 0.91,
     "snippet": "Troponin levels exceeding the 99th percentile of the upper reference limit indicate myocardial injury..."},
    {"source": "FDA NSAID Cardiovascular Risk Warning", "relevance_score": 0.87,
     "snippet": "NSAIDs cause an increased risk of serious cardiovascular thrombotic events, including myocardial infarction and stroke..."}
  ],
  "safety_flags": [
    {"level": "critical",
     "message": "Aspirin contraindicated — patient allergy (anaphylaxis). Clopidogrel substituted.",
     "related_recommendation": "Start Clopidogrel 300mg loading dose"},
    {"level": "warning",
     "message": "NSAID use (6 months) associated with elevated cardiovascular risk. Discontinuation recommended.",
     "related_recommendation": "Discontinue NSAIDs"}
  ],
  "constraint_violations": [
    {"rule": "ALLERGY_CONTRAINDICATION",
     "description": "Proposed drug (Aspirin) matches patient allergy (Aspirin → anaphylaxis, severe)",
     "action_blocked": "Start Aspirin 325mg"}
  ],
  "confidence": {
    "overall": 0.85,
    "evidence_strength": 0.90,
    "model_certainty": 0.78,
    "guideline_compliance": 0.88
  },
  "requires_human_review": true,
  "escalation_reason": "STAT medication recommendations require mandatory clinician sign-off"
}
```

---

## 9. Phase 0: Problem Validation (Week 1–2)

### 9.1 Objective
Validate that the system concept produces useful clinical reasoning before building infrastructure.

### 9.2 What Gets Built

| Component | Implementation |
|-----------|---------------|
| **LLM Setup** | Agno with single `Agent`, Ollama gpt-oss:20b / HuggingFace |
| **Input** | Manual JSON via Streamlit text area |
| **Prompt** | Structured clinical reasoning prompt (v0.1) |
| **Output** | Raw JSON response displayed in Streamlit |
| **Evaluation** | Manual review of 10 MedQA cases + 5 synthetic patient cases |

### 9.3 Deliverables

1. `src/nstea/agents/reasoning_agent.py` — Single Agent with medical system prompt
2. `src/nstea/agents/prompts/reasoning.py` — v0.1 structured reasoning prompt
3. `frontend/streamlit_app/app.py` — Minimal input/output interface
4. `notebooks/01_data_exploration.ipynb` — MedQA test results documentation

### 9.4 Decision Gate

| Metric | Threshold | Action if Failed |
|--------|-----------|-----------------|
| MedQA accuracy (10 cases) | > 60% | Revise system prompt |
| Clinical relevance (expert review) | > 70% useful | Proceed to Phase 1 |
| Hallucination rate | < 30% | If exceeded, add RAG immediately |

### 9.5 Files Changed / Created
```
src/nstea/__init__.py
src/nstea/config.py
src/nstea/agents/__init__.py
src/nstea/agents/reasoning_agent.py
src/nstea/agents/prompts/__init__.py
src/nstea/agents/prompts/reasoning.py
frontend/streamlit_app/app.py
pyproject.toml
```

---

## 10. Phase 1: MVP (RAG + LLM + Rule Engine) — Week 3–6

### 10.1 Objective
Build a usable clinical decision support system with grounded recommendations and hard safety rules.

### 10.2 Component Breakdown

#### 10.2.1 Retrieval Layer (RAG)

| Step | Implementation | Tech |
|------|---------------|------|
| Document Loading | Chunk clinical guidelines (PDF/text) into 512-token segments with 64-token overlap | LangChain text splitter |
| Embedding | BioClinicalBERT or `text-embedding-004` (Google) | sentence-transformers / Google API |
| Vector Storage | Index chunks in Qdrant (local Docker instance) | qdrant-client |
| Retrieval | Cosine similarity search, Top-5 with score threshold ≥ 0.7 | Custom retriever class |
| Context Assembly | Combine: patient summary + retrieved docs + clinician query | context_builder.py |

Initial knowledge base:
- ACC/AHA Clinical Guidelines (Cardiology)
- WHO Essential Medicines List
- Common drug interaction databases (public)
- Standard lab reference ranges

#### 10.2.2 Rule Engine (Hard Safety)

```yaml
# data/rules/contraindications.yml
rules:
  - id: ALLERGY_DRUG_MATCH
    description: "Block any drug matching a known patient allergy"
    type: hard_block
    condition: "proposed_drug IN patient_allergies"
    action: reject
    severity: critical

  - id: DRUG_INTERACTION_SEVERE
    description: "Block known severe drug-drug interactions"
    type: hard_block
    condition: "interaction_severity == 'severe'"
    action: reject
    severity: critical

  - id: RENAL_DOSING
    description: "Flag medications requiring renal adjustment when eGFR < 30"
    type: flag
    condition: "drug.requires_renal_adjustment AND patient.eGFR < 30"
    action: warn
    severity: warning
    
  - id: PREGNANCY_CATEGORY_X
    description: "Block Category X drugs in pregnant patients"
    type: hard_block
    condition: "drug.pregnancy_category == 'X' AND patient.is_pregnant"
    action: reject
    severity: critical
```

#### 10.2.3 LLM Orchestrator (Agno)

Phase 1 uses a simplified pipeline (no T-GNN, no parallel enrichment):

```
Workflow (sequential):
  1. RetrievalAgent → RAG context
  2. Workflow (loop pattern, max 3):
       a. ReasoningAgent (Agent + pre-computed context)
       b. SafetyAgent (rule engine)
       c. CheckStatus
  3. ConfidenceAgent → final output
```

#### 10.2.4 API Layer

| Endpoint | Implementation |
|----------|---------------|
| `POST /api/v1/analyze` | Async analysis submission (returns 202 + analysis_id) |
| `GET /api/v1/analyze/{id}` | Poll for results |
| `POST /api/v1/patients` | Register patient data |
| `GET /api/v1/health` | Basic health check |

#### 10.2.5 Streamlit MVP UI

- Patient input form (structured fields)
- Analysis trigger button
- Formatted recommendation display
- Safety alerts section
- Basic reasoning trace

### 10.3 Deliverables

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| Working RAG pipeline | Retrieves relevant guidelines for cardiology + common medicine cases |
| Rule engine | Catches aspirin + allergy, drug-drug interactions, renal dosing |
| Agno agent pipeline | End-to-end: input → RAG → LLM → safety → output |
| FastAPI backend | All 4 endpoints functional |
| Streamlit UI | Clinician can input patient + see recommendation |
| Latency | < 5 seconds for simple cases |

### 10.4 Files Changed / Created
```
# Data Layer
src/nstea/data_layer/preprocessor.py
src/nstea/data_layer/schema_mapper.py

# Retrieval
src/nstea/retrieval/embedder.py
src/nstea/retrieval/vector_store.py
src/nstea/retrieval/retriever.py
src/nstea/retrieval/context_builder.py
src/nstea/retrieval/document_loader.py

# Agents
src/nstea/agents/orchestrator.py         # Simplified Phase 1 pipeline
src/nstea/agents/retrieval_agent.py
src/nstea/agents/safety_agent.py
src/nstea/agents/confidence_agent.py

# Tools
src/nstea/tools/guideline_lookup.py
src/nstea/tools/drug_interaction.py
src/nstea/tools/contraindication.py

# Safety
src/nstea/safety/rule_engine.py
data/rules/contraindications.yml
data/rules/drug_interactions.yml

# API
src/nstea/main.py
src/nstea/api/routes/analysis.py
src/nstea/api/routes/patients.py
src/nstea/api/deps.py

# Models
src/nstea/models/patient.py
src/nstea/models/analysis.py
src/nstea/models/recommendation.py

# Services
src/nstea/services/analysis_service.py
src/nstea/services/patient_service.py

# Core
src/nstea/core/database.py
src/nstea/core/exceptions.py
src/nstea/core/logging.py

# Frontend
frontend/streamlit_app/app.py (updated)
frontend/streamlit_app/pages/patient_input.py
frontend/streamlit_app/pages/analysis.py

# Infrastructure
docker-compose.yml                        # PostgreSQL + Qdrant + Redis
scripts/index_guidelines.py
scripts/generate_synthetic_data.py
data/guidelines/                          # Initial guideline documents
data/synthetic/patients.json

# Tests
tests/unit/test_preprocessor.py
tests/unit/test_retriever.py
tests/unit/test_rule_engine.py
tests/integration/test_analysis_pipeline.py
```

---

## 11. Phase 2: Production MVP (Systemization) — Week 7–10

### 11.1 Objective
Make the system stable, observable, testable, and deployable.

### 11.2 Component Breakdown

#### 11.2.1 Orchestration Hardening

| Enhancement | Implementation |
|-------------|---------------|
| **Retry Logic** | Agno Workflow (loop) for safety retries; exponential backoff for external API calls |
| **Timeout Management** | 30s max per analysis; graceful degradation if T-GNN/RAG times out |
| **Error Boundaries** | Each agent wrapped with try/except; partial results returned on failure |
| **State Tracking** | Full Agno session state tracking with event history |

#### 11.2.2 Caching Layer

```python
# Redis caching strategy
CACHE_STRATEGY = {
    "guideline_embeddings": {
        "backend": "Redis",
        "ttl": "7 days",        # Guidelines don't change frequently
        "invalidation": "manual (on re-index)"
    },
    "rag_results": {
        "backend": "Redis",
        "ttl": "1 hour",        # Same query → same results (short-term)
        "key": "hash(query + patient_context)"
    },
    "tgnn_embeddings": {        # Phase 4 prep
        "backend": "Redis",
        "ttl": "24 hours",
        "key": "patient_id + data_hash"
    },
    "analysis_results": {
        "backend": "PostgreSQL",
        "ttl": "permanent",     # Audit log — never delete
        "key": "analysis_id"
    }
}
```

#### 11.2.3 Monitoring & Observability

| Metric | Tool | Alert Threshold |
|--------|------|----------------|
| API latency (P50, P95, P99) | Prometheus + Grafana | P95 > 10s |
| LLM token usage per request | Custom counter | > 50K tokens/request |
| Safety violation rate | Custom counter | > 15% of analyses |
| RAG retrieval relevance | Custom histogram | Mean < 0.6 |
| Error rate | Prometheus | > 5% of requests |
| Cache hit rate | Redis metrics | < 30% |

#### 11.2.4 Evaluation Harness

```python
# Batch evaluation pipeline
# scripts/run_evaluation.py

EVALUATION_SUITE = {
    "medqa_accuracy": {
        "dataset": "data/eval/medqa_sample.json",  # 100 cases
        "metric": "accuracy",
        "target": "> 65%"
    },
    "safety_violation_detection": {
        "dataset": "data/eval/safety_cases.json",   # 50 known-dangerous cases
        "metric": "recall (violations caught)",
        "target": "> 95%"
    },
    "hallucination_rate": {
        "dataset": "data/eval/adversarial.json",    # 30 trick questions
        "metric": "hallucination_rate",
        "target": "< 20%"
    },
    "latency": {
        "dataset": "data/eval/latency_cases.json",  # 50 varied complexity
        "metric": "P95 latency",
        "target": "< 8 seconds"
    }
}
```

#### 11.2.5 Database + Audit

- Full analysis audit logging (every request + response + reasoning trace)
- Alembic migration pipeline functional
- Feedback table populated

### 11.3 Deliverables

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| Redis caching | RAG cache hit rate > 40% on repeat queries |
| Prometheus + Grafana | Dashboard showing latency, errors, safety metrics |
| Structured logging | All agent steps logged with correlation ID |
| Evaluation harness | Automated batch evaluation with report generation |
| Docker Compose | Full local dev stack: API + DB + Qdrant + Redis + Grafana |
| CI pipeline | GitHub Actions: lint + unit tests + safety tests on every PR |

### 11.4 Key New Files
```
# Monitoring
src/nstea/core/metrics.py
src/nstea/api/middleware.py

# Caching
src/nstea/core/redis_client.py

# Evaluation
scripts/run_evaluation.py
data/eval/medqa_sample.json
data/eval/safety_cases.json
data/eval/adversarial.json
tests/evaluation/eval_medqa.py
tests/evaluation/eval_safety.py
tests/evaluation/eval_latency.py

# Infrastructure
infra/docker/Dockerfile.backend
docker-compose.yml (updated with Prometheus, Grafana)
.github/workflows/ci.yml

# Database
src/nstea/models/db/base.py
src/nstea/models/db/analysis_log.py
src/nstea/models/db/feedback.py
alembic/
```

---

## 12. Phase 3: Structured Reasoning Upgrade — Week 11–13

### 12.1 Objective
Reduce hallucination, improve diagnostic reliability, add confidence gating.

### 12.2 Component Breakdown

#### 12.2.1 Structured Reasoning Templates

```python
# Reasoning prompt structure (injected into Agent instruction)

REASONING_TEMPLATE = """
You are a clinical reasoning engine. Follow this EXACT structure:

## STEP 1: PATIENT SUMMARY
Summarize key facts: demographics, active conditions, current medications,
allergies, presenting symptoms, abnormal labs.

## STEP 2: PROBLEM IDENTIFICATION
List the clinical problems that need addressing, ordered by acuity.

## STEP 3: DIFFERENTIAL DIAGNOSIS
For EACH problem, generate a differential diagnosis:
- Diagnosis name
- Supporting evidence (from patient data)
- Contradicting evidence
- Estimated probability (0.0–1.0)

## STEP 4: EVIDENCE MATCHING
For each leading diagnosis, cite the specific retrieved guideline that supports it.
If NO guideline supports it, state: "NO GUIDELINE SUPPORT — low confidence."

## STEP 5: TREATMENT PLAN
For each diagnosis (starting with most likely):
- Recommended action
- Category (medication/test/procedure/referral)
- Urgency (stat/urgent/routine)
- Rationale (cite guideline)

## STEP 6: SAFETY SELF-CHECK
Before finalizing, verify:
- No proposed drug matches patient allergies
- No known drug-drug interactions with current medications
- Dosing appropriate for patient's renal/hepatic function

If you find ANY safety concern, flag it explicitly.
"""
```

#### 12.2.2 Tool Usage Enhancement

| New Tool | Purpose |
|----------|---------|
| `lab_calculator` | eGFR, CHA₂DS₂-VASc, MELD, Wells Score, CURB-65 |
| `dosage_calculator` | Renal/hepatic-adjusted dosing |
| Enhanced `guideline_lookup` | Multi-query RAG with re-ranking |

#### 12.2.3 Confidence Scoring (Implemented)

```python
class ConfidenceScorer:
    """Calculates multi-dimensional confidence score."""
    
    def calculate(
        self,
        retrieved_docs: list[dict],
        reasoning_trace: ReasoningTrace,
        safety_result: str,
    ) -> ConfidenceScore:
        
        # Evidence strength: average relevance of top-3 retrieved docs
        evidence_strength = mean([d["relevance_score"] for d in retrieved_docs[:3]])
        
        # Model certainty: does the LLM express uncertainty markers?
        # Parse reasoning for: "uncertain", "possible", "consider" vs "definitive", "clear"
        model_certainty = self._assess_linguistic_certainty(reasoning_trace)
        
        # Guideline compliance: what % of recommendations cite a guideline?
        guideline_compliance = self._check_guideline_citations(reasoning_trace)
        
        overall = (
            0.40 * evidence_strength +
            0.25 * model_certainty +
            0.35 * guideline_compliance
        )
        
        return ConfidenceScore(
            overall=overall,
            evidence_strength=evidence_strength,
            model_certainty=model_certainty,
            guideline_compliance=guideline_compliance
        )

# Gating logic
CONFIDENCE_THRESHOLD = 0.75
if confidence.overall < CONFIDENCE_THRESHOLD:
    result.requires_human_review = True
    result.escalation_reason = f"Confidence {confidence.overall:.0%} below threshold"
```

### 12.3 Deliverables

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| Structured reasoning | All outputs follow 6-step template |
| Clinical calculators | 5+ calculators functional as Agno tools |
| Confidence scoring | Score computed for every analysis |
| HITL gating | Low-confidence cases flagged for human review |
| Hallucination reduction | < 15% on adversarial test set (down from ~25%) |
| Feedback UI | Clinicians can Accept/Modify/Reject + add notes |

### 12.4 Key New/Updated Files
```
# Updated prompts
src/nstea/agents/prompts/reasoning.py     # Structured template v1.0
src/nstea/agents/prompts/safety.py

# New tools
src/nstea/tools/lab_calculator.py

# Confidence
src/nstea/agents/confidence_agent.py      # Full implementation

# Feedback
src/nstea/api/routes/feedback.py
src/nstea/services/feedback_service.py
src/nstea/models/feedback.py
frontend/streamlit_app/pages/feedback.py

# Tests
tests/safety/test_hallucination.py
tests/safety/test_contraindications.py
```

---

## 13. Phase 4: Temporal Layer (T-GNN Lite) — Week 14–18

### 13.1 Objective
Introduce temporal awareness via batch-computed graph embeddings, NOT real-time.

### 13.2 Critical Design Decision

```
❌ WILL NOT BUILD: Real-time dynamic graph construction
✅ WILL BUILD: Batch graph construction + cached embeddings

Rationale:
- Real-time T-GNN adds 5-10s latency per request
- EHR data is noisy, asynchronous, incomplete
- Batch updates (daily/on-data-change) are sufficient for clinical relevance
- Embeddings cached in Redis, served in <50ms
```

### 13.3 Component Breakdown

#### 13.3.1 Graph Builder

```python
class PatientGraphBuilder:
    """Converts patient clinical history → temporal graph."""
    
    def build_graph(self, patient: PatientInput) -> TemporalGraph:
        G = nx.DiGraph()
        
        # Add nodes: each clinical event
        for event in patient.history:
            G.add_node(event.id, **{
                "type": event.event_type,         # diagnosis, medication, lab...
                "description": event.description,
                "date": event.date,
                "features": self._extract_features(event),
            })
        
        # Add edges: temporal + causal
        events_sorted = sorted(patient.history, key=lambda e: e.date)
        for i in range(len(events_sorted) - 1):
            current = events_sorted[i]
            next_event = events_sorted[i + 1]
            time_gap = (next_event.date - current.date).days
            
            G.add_edge(current.id, next_event.id, **{
                "type": "temporal",
                "time_gap_days": time_gap,
                "weight": math.exp(-self.decay_lambda * time_gap),
            })
        
        # Add causal edges (known relationships)
        self._add_causal_edges(G, patient)
        
        return TemporalGraph(networkx_graph=G, patient_id=patient.patient_id)
    
    def _add_causal_edges(self, G, patient):
        """Add edges for known causal relationships.
        e.g., NSAID use → cardiovascular risk"""
        # Uses ontology mappings from knowledge base
        ...
```

#### 13.3.2 Temporal Encoder

```python
class TemporalEncoder:
    """Computes time-decay attention weights for graph nodes."""
    
    def __init__(self, decay_lambda: float = 0.01):
        self.decay_lambda = decay_lambda
    
    def encode(self, graph: TemporalGraph, reference_date: date) -> dict:
        """Compute importance weights for each node relative to reference date."""
        weights = {}
        for node_id, data in graph.G.nodes(data=True):
            time_gap = (reference_date - data["date"]).days
            weight = math.exp(-self.decay_lambda * time_gap)
            weights[node_id] = {
                "temporal_weight": weight,
                "features": data["features"],
                "weighted_features": data["features"] * weight
            }
        return weights
```

#### 13.3.3 T-GNN Model (PyTorch Geometric)

```python
class TemporalGNN(torch.nn.Module):
    """Lightweight T-GNN for patient embedding generation."""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=2)
        self.lin = Linear(hidden_channels * 2, out_channels)
        self.time_encoder = TimeEncoder(hidden_channels)
    
    def forward(self, x, edge_index, edge_attr, timestamps):
        # Encode temporal information
        time_emb = self.time_encoder(timestamps)
        x = x + time_emb  # Inject temporal awareness
        
        # Graph convolution with attention
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr))
        
        # Global pooling → patient embedding
        x = global_mean_pool(x, batch=None)
        x = self.lin(x)
        return x  # Dense patient embedding vector (e.g., 256-dim)
```

#### 13.3.4 Batch Update Strategy

```python
# Batch scheduler: runs daily or on-data-change

class BatchEmbeddingUpdater:
    """Scheduled batch computation of T-GNN embeddings."""
    
    async def update_all(self):
        """Recompute embeddings for patients with new data."""
        patients_with_updates = await self.db.get_patients_with_new_data(
            since=self.last_run_timestamp
        )
        
        for patient in patients_with_updates:
            graph = self.graph_builder.build_graph(patient)
            embedding = self.tgnn_model(graph.to_pyg())
            insights = self.generate_temporal_insights(graph)
            
            # Cache in Redis with 24h TTL
            await self.cache.set(
                key=f"tgnn:{patient.patient_id}:{patient.data_hash}",
                value={"embedding": embedding.tolist(), "insights": insights},
                ttl=86400
            )
    
    def generate_temporal_insights(self, graph) -> list[str]:
        """Generate human-readable temporal insights from graph."""
        insights = []
        # Example: "NSAID use started 6 months ago → elevated CV risk"
        # Parse high-weight edges and causal relationships
        ...
        return insights
```

#### 13.3.5 Integration with Agno Pipeline

The `TemporalAgent` runs in parallel with `RetrievalAgent`:

```python
class TemporalAgent(BaseAgent):
    """Fetches or computes T-GNN embedding for patient."""
    
    async def _run_async_impl(self, ctx):
        patient_data = ctx.session.state.get("patient_data")
        patient_id = patient_data["patient_id"]
        data_hash = compute_hash(patient_data["history"])
        
        # Try cache first
        cached = await self.cache.get(f"tgnn:{patient_id}:{data_hash}")
        
        if cached:
            embedding = cached["embedding"]
            insights = cached["insights"]
        else:
            # Compute on-the-fly (fallback; slower)
            graph = self.graph_builder.build_graph(patient_data)
            embedding = self.tgnn_model(graph.to_pyg())
            insights = self.generate_insights(graph)
        
        ctx.session.state["temp:temporal_embedding"] = embedding
        ctx.session.state["temp:temporal_insights"] = insights
        
        yield Event(
            author=self.name,
            content=Content(parts=[Part(text=f"Temporal insights: {insights}")])
        )
```

### 13.4 Deliverables

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| Graph builder | Converts any patient history → NetworkX graph |
| T-GNN model | Generates 256-dim embeddings; trainable |
| Batch updater | Celery task runs daily; updates Redis cache |
| TemporalAgent | Integrated into Agno pipeline (parallel with RAG) |
| Temporal insights | Human-readable insights shown in UI |
| Cytoscape.js viz | Interactive temporal graph in frontend (Phase 4 only) |
| Latency | < 50ms for cached embedding retrieval |

### 13.5 Key New Files
```
# Temporal Engine
src/nstea/temporal/__init__.py
src/nstea/temporal/graph_builder.py
src/nstea/temporal/temporal_encoder.py
src/nstea/temporal/tgnn_model.py
src/nstea/temporal/embedding_cache.py
src/nstea/temporal/batch_updater.py

# Agent
src/nstea/agents/temporal_agent.py

# Training
mlops/training/train_tgnn.py
notebooks/03_tgnn_prototype.ipynb

# Frontend
frontend/web/src/components/TemporalGraph.tsx

# Updated orchestrator
src/nstea/agents/orchestrator.py    # Add Team with TemporalAgent
```

---

## 14. Phase 5: Symbolic Constraint Engine — Week 19–22

### 14.1 Objective
Replace YAML rule engine with Neo4j-backed formal symbolic constraint engine.

### 14.2 Knowledge Graph Design (Neo4j)

```cypher
// Node types
(:Drug {name, rxnorm_code, category, pregnancy_category})
(:Condition {name, icd10_code, category})
(:Allergy {substance, severity})
(:Interaction {severity, description, mechanism})
(:Contraindication {type, description, severity})
(:Guideline {name, source, version, url})

// Relationship types
(:Drug)-[:INTERACTS_WITH {severity, description}]->(:Drug)
(:Drug)-[:CONTRAINDICATED_IN]->(:Condition)
(:Drug)-[:CAUSES_ALLERGY_IN]->(:Allergy)
(:Condition)-[:REQUIRES_CAUTION_WITH]->(:Drug)
(:Guideline)-[:RECOMMENDS {strength}]->(:Drug)
(:Guideline)-[:FOR_CONDITION]->(:Condition)
```

### 14.3 Constraint Validation Logic

```python
class SymbolicConstraintEngine:
    """Neo4j-backed formal constraint validator."""
    
    async def validate(
        self,
        proposed_actions: list[Recommendation],
        patient: PatientInput
    ) -> ConstraintResult:
        
        violations = []
        
        for action in proposed_actions:
            if action.category == "medication":
                drug = action.parsed_drug_name
                
                # Check 1: Allergy match
                allergy_violations = await self._check_allergies(drug, patient.allergies)
                violations.extend(allergy_violations)
                
                # Check 2: Drug-drug interactions
                interaction_violations = await self._check_interactions(
                    drug, [m.name for m in patient.medications]
                )
                violations.extend(interaction_violations)
                
                # Check 3: Condition contraindications
                condition_violations = await self._check_condition_contraindications(
                    drug, [c.name for c in patient.conditions]
                )
                violations.extend(condition_violations)
        
        # Formal constraint: Condition ∩ Contraindication = ∅
        is_safe = len([v for v in violations if v.severity == "critical"]) == 0
        
        return ConstraintResult(
            is_safe=is_safe,
            violations=violations,
            unmapped_actions=self._find_unmapped(proposed_actions)  # Honest about gaps
        )
    
    async def _check_allergies(self, drug: str, allergies: list) -> list:
        """Cypher query: Does this drug match any patient allergy?"""
        query = """
        MATCH (d:Drug {name: $drug})-[:CAUSES_ALLERGY_IN]->(a:Allergy)
        WHERE a.substance IN $allergy_substances
        RETURN d.name as drug, a.substance as allergy, a.severity as severity
        """
        results = await self.neo4j.run(query, drug=drug,
                                        allergy_substances=[a.substance for a in allergies])
        return [ConstraintViolation(...) for r in results]
```

### 14.4 Key Principle: Honest About Incompleteness

```python
class ConstraintResult(BaseModel):
    is_safe: bool
    violations: list[ConstraintViolation]
    unmapped_actions: list[str]  # Actions NOT covered by any constraint rule
    coverage_percentage: float    # What % of proposed actions were fully validated
    
    # WARNING: If unmapped_actions is non-empty, the system explicitly tells the clinician:
    # "The following actions could not be validated against our knowledge base.
    #  Human review is MANDATORY."
```

### 14.5 Deliverables

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| Neo4j knowledge graph | SNOMED-CT + RxNorm + drug interactions loaded |
| Symbolic constraint engine | Validates allergies, interactions, contraindications |
| Honest incompleteness | Reports unmapped actions with mandatory HITL |
| Feedback loop to KG | Clinician corrections can suggest new rules |
| Replaces rule engine | All Phase 1 YAML rules migrated to Neo4j |

### 14.6 Key New Files
```
src/nstea/safety/constraint_engine.py     # Neo4j-backed engine
src/nstea/safety/knowledge_graph.py       # Neo4j client + queries
src/nstea/core/neo4j_client.py
scripts/seed_knowledge_graph.py           # Load SNOMED/RxNorm into Neo4j

data/ontologies/snomed_ct_subset.csv
data/ontologies/rxnorm_interactions.csv
```

---

## 15. Phase 6: System Hardening — Week 23–26

### 15.1 Components

#### 15.1.1 Security

| Control | Implementation |
|---------|---------------|
| Authentication | OAuth2/OIDC (Google Identity / Keycloak) |
| Authorization | Role-Based Access Control (RBAC): admin, clinician, read-only |
| Encryption at rest | PostgreSQL TDE; encrypted Redis |
| Encryption in transit | TLS 1.3 everywhere |
| API rate limiting | Redis-based token bucket (60 req/min per user) |
| Input sanitization | Pydantic validators + custom sanitizers |
| Audit trail | Every action logged with user ID + timestamp |

#### 15.1.2 Data Drift Detection

```python
class DriftDetector:
    """Monitors incoming patient data distributions."""
    
    def check_drift(self, recent_data: pd.DataFrame, baseline: pd.DataFrame) -> DriftReport:
        # Jensen-Shannon divergence for each feature
        for feature in self.monitored_features:
            js_div = jensenshannon(
                recent_data[feature].value_counts(normalize=True),
                baseline[feature].value_counts(normalize=True)
            )
            if js_div > self.threshold:
                self.trigger_alert(feature, js_div)
```

#### 15.1.3 Production Frontend (Next.js)

Full production UI with all 4 screens (Section 6.2):
- Dashboard, Patient Input, Analysis Results, Admin
- WebSocket integration for real-time reasoning stream
- Responsive design (tablet-compatible for hospital use)
- WCAG 2.1 AA accessibility

#### 15.1.4 HITL Dashboard Enhancement

- Override tracking (every human modification logged)
- Analytics: acceptance rate by diagnosis category
- Alert fatigue monitoring (track time-to-decision)

### 15.2 Deliverables

| Deliverable | Acceptance Criteria |
|-------------|-------------------|
| Auth system | OAuth2 login; RBAC enforced on all endpoints |
| Drift detection | Alerts on demographic/clinical data shifts |
| Next.js production UI | All 4 screens functional with real data |
| Audit system | Complete traceability: every decision auditable |
| Load testing | Handles 50 concurrent analyses without degradation |

---

## 16. Phase 7: Advanced Features (Optional)

Only after Phase 6 stability is proven:

| Feature | Implementation | Risk |
|---------|---------------|------|
| **Multi-Agent Debate** | Add Team with 2-3 specialist Agents (Cardiology, Pharmacy) that independently analyze, then a Workflow synthesizes | Token explosion; manage with strict output limits |
| **RL Self-Improvement** | Offline DPO training on clinician feedback data | Requires significant feedback volume (>1000 cases) |
| **Simulation Mode** | Generate synthetic patients; run through pipeline; auto-evaluate | Agent Hospital–style; useful for continuous improvement |
| **FHIR Integration** | Real FHIR endpoint connection (read-only) | Requires hospital IT partnership |

---

## 17. Testing Strategy

### 17.1 Test Pyramid

```
                    ┌──────────────┐
                    │  E2E Tests   │    5 critical user journeys
                    │  (Playwright)│    Run: pre-deploy
                    ├──────────────┤
                  ┌──────────────────┐
                  │  Integration     │    Agent pipeline, API endpoints,
                  │  Tests (pytest)  │    DB queries. Run: every PR
                  ├──────────────────┤
                ┌──────────────────────┐
                │  Unit Tests (pytest) │    Every function, every model,
                │  + Safety Tests      │    every rule. Run: every commit
                └──────────────────────┘
```

### 17.2 Safety-Specific Test Suite

```python
# tests/safety/test_contraindications.py

KNOWN_DANGEROUS_CASES = [
    # (patient_allergies, proposed_drug, expected: BLOCK)
    (["Aspirin"], "Aspirin", True),
    (["Penicillin"], "Amoxicillin", True),       # Cross-reactivity
    (["Sulfa drugs"], "Sulfamethoxazole", True),
    (["NSAIDs"], "Ibuprofen", True),
]

KNOWN_INTERACTION_CASES = [
    # (current_meds, proposed_drug, expected: at least WARNING)
    (["Warfarin"], "Aspirin", True),
    (["MAO inhibitor"], "SSRI", True),
    (["Methotrexate"], "NSAID", True),
    (["Lithium"], "ACE inhibitor", True),
]

# Every PR MUST pass ALL safety test cases.
# Adding a new case requires review from clinical advisor.
```

### 17.3 Evaluation Benchmarks

| Benchmark | Frequency | Target |
|-----------|-----------|--------|
| MedQA accuracy (100 cases) | Weekly | > 65% |
| Safety violation recall | Every PR | > 95% |
| Hallucination rate (adversarial) | Weekly | < 15% |
| Latency P95 | Every PR | < 10s |
| VLY (Verifiable Logic Yield) | Monthly | > 70% |

---

## 18. Deployment Strategy

### 18.1 Local Development

```yaml
# docker-compose.yml provides:
services:
  backend:     # FastAPI app
  postgres:    # PostgreSQL 16
  qdrant:      # Qdrant vector DB
  redis:       # Redis 7
  neo4j:       # Neo4j (Phase 5+)
  prometheus:  # Metrics
  grafana:     # Dashboards
```

### 18.2 Staging → Production

| Stage | Environment | Purpose |
|-------|------------|---------|
| **Local** | Docker Compose | Development + unit tests |
| **CI** | GitHub Actions | Automated testing on every PR |
| **Staging** | GKE (single-node) | Integration testing with real data |
| **Shadow** | GKE (production mirror) | 90-day shadow mode: generate recommendations but DON'T expose to clinicians. Log everything for offline comparison. |
| **Production** | GKE (multi-node, hardened) | Live clinical use with mandatory HITL |

### 18.3 Kubernetes Architecture

```yaml
# Namespace: nstea-production
Pods:
  - backend (3 replicas, autoscale 3-10)
  - celery-worker (2 replicas)         # Batch T-GNN updates
  - frontend (2 replicas)
  - qdrant (1 replica, persistent volume)
  - redis (1 replica, persistent volume)
  - neo4j (1 replica, persistent volume) # Phase 5+
  - postgres (managed Cloud SQL)

Ingress:
  - HTTPS only (TLS 1.3)
  - Cloud Armor WAF
  
Resources per backend pod:
  CPU: 2 cores
  Memory: 4Gi
  GPU: None (LLM via API; T-GNN model is lightweight)
```

---

## 19. Security & Compliance

### 19.1 HIPAA Alignment

| HIPAA Requirement | NS-TEA Implementation |
|---|---|
| Access Controls | RBAC with OAuth2/OIDC; minimum necessary access |
| Audit Controls | Every API call, agent decision, and data access logged |
| Integrity Controls | Read-only EHR access; no write operations |
| Transmission Security | TLS 1.3 for all connections |
| Encryption | AES-256 at rest; encrypted Redis; encrypted backups |
| Business Associate Agreement | Required for cloud LLM API usage (if applicable) |

### 19.2 Adversarial Protection

| Threat | Mitigation |
|--------|-----------|
| Prompt injection via patient data | Input sanitization; system prompt isolation via Agno guardrails |
| PHI leakage through LLM API | Strip unnecessary PHI before sending to LLM; use anonymized IDs |
| Data exfiltration | No egress from agent sandbox; network policies in K8s |
| Model manipulation | Pinned model versions; drift detection |

---

## 20. Risk Matrix & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| LLM hallucination causes harmful recommendation | High | Critical | RAG grounding + constraint engine + mandatory HITL |
| T-GNN generates incorrect temporal insight | Medium | High | Batch mode (human-reviewable); temporal insights are advisory only |
| Knowledge graph incompleteness misses contraindication | Medium | Critical | Explicit "unmapped actions" reporting; mandatory HITL for unmapped |
| Latency exceeds budget under load | Medium | Medium | Caching, autoscaling, graceful degradation, timeout management |
| LLM API / Ollama downtime | Low | High | Retry with backoff; fallback to cached recommendations; clear error messaging |
| Clinician alert fatigue (too many escalations) | Medium | High | Tune confidence threshold; track escalation rate; target < 10% escalation |
| Demographic bias in recommendations | Medium | High | Bias test suite; monitor demographic parity in outcomes |
| Adversarial prompt injection | Low | Critical | Input sanitization; Agno guardrails for prompt monitoring; WAF |

---

## 21. Timeline & Milestones

```
Week  1-2   ┃ Phase 0: Problem Validation
            ┃ → Decision Gate: Proceed or revise prompts
            ┃
Week  3-6   ┃ Phase 1: MVP (RAG + LLM + Rules)
            ┃ → Milestone: Working clinical assistant (Streamlit)
            ┃
Week  7-10  ┃ Phase 2: Production MVP
            ┃ → Milestone: Stable, monitored, testable system
            ┃
Week 11-13  ┃ Phase 3: Structured Reasoning
            ┃ → Milestone: Confidence scoring + HITL gating live
            ┃
Week 14-18  ┃ Phase 4: Temporal Layer (T-GNN Lite)
            ┃ → Milestone: Time-aware patient embeddings in pipeline
            ┃
Week 19-22  ┃ Phase 5: Symbolic Constraint Engine
            ┃ → Milestone: Neo4j-backed formal safety validation
            ┃
Week 23-26  ┃ Phase 6: System Hardening
            ┃ → Milestone: Production-ready with auth, audit, drift detection
            ┃
Week 27+    ┃ Phase 7: Advanced Features (optional)
            ┃ → Multi-agent debate, RL, FHIR integration
```

**Total: 26 weeks (6.5 months) to production-ready system**

---

## Approval Checklist

Before implementation begins, confirm:

- [ ] Tech stack approved (Python + Agno + Ollama/HuggingFace + Qdrant + Neo4j + Next.js)
- [ ] Phase 0 scope approved (10 MedQA cases + 5 synthetic patients)
- [ ] Data models approved (PatientInput, AnalysisResponse, Feedback)
- [ ] API contracts approved (REST endpoints + WebSocket)
- [ ] UI/UX wireframes approved (4 screens)
- [ ] Agno agent hierarchy approved (Workflow → Team → Workflow loop)
- [ ] Safety approach approved (rule engine → Neo4j constraint engine evolution)
- [ ] Deployment strategy approved (Docker → GKE with shadow mode)
- [ ] Timeline approved (26 weeks, phased)

---

**No code will be written until this plan receives explicit approval.**
