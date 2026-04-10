<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Tests-104%20Passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/Phase-5%20Complete-blueviolet" alt="Phase"/>
  <img src="https://img.shields.io/badge/License-Research%20Only-orange" alt="License"/>
</p>

<h1 align="center">NS-TEA: Neuro-Symbolic Temporal EHR Agent</h1>
<h3 align="center">AI-Powered Clinical Decision Support System with Evidence-Based Reasoning</h3>

<p align="center">
  <em>A hybrid neuro-symbolic system combining Large Language Models, Retrieval-Augmented Generation,<br/>
  Temporal Patient Graphs, and Symbolic Constraint Engines for safe, explainable clinical reasoning.</em>
</p>

---

## TABLE OF CONTENTS

| Chapter | Section | Description |
|---------|---------|-------------|
| **Chapter 1** | [Introduction](#chapter-1--introduction) | Project overview, problem statement, aims, features, scope |
| | [1.1 Introduction](#11-introduction) | What is NS-TEA and why it matters |
| | [1.2 Problem Statement](#12-problem-statement) | Challenges in clinical AI |
| | [1.3 Aims and Objectives](#13-aims-and-objectives) | Project goals and deliverables |
| | [1.4 Features](#14-features) | Core capabilities |
| | [1.5 Future Scope](#15-future-scope) | Long-term vision |
| **Chapter 2** | [Project Plan](#chapter-2--project-plan) | Technical stack, architecture, workflow |
| | [2.1 Introduction](#21-introduction) | Development methodology |
| | [2.2 System Architecture](#22-system-architecture) | Architecture diagrams and data flow |
| | [2.3 Workflow](#23-workflow) | End-to-end pipeline flow |
| | [2.4 Tools Used](#24-tools-used) | Development toolchain |
| | [2.5 Platform Used in Development](#25-platform-used-in-development) | OS, runtime, IDE |
| | [2.6 Libraries Used](#26-libraries-used) | All dependencies explained |
| **Chapter 3** | [Working of Project](#chapter-3--working-of-project) | Implementation details |
| | [3.1 Research and Requirements Gathering](#31-research-and-requirements-gathering) | Literature review and design |
| | [3.2 Initialization and Setup](#32-initialization-and-setup) | Environment and project setup |
| | [3.3 Implementation](#33-implementation) | Phase-by-phase build details |
| **Chapter 4** | [Conclusion](#chapter-4--conclusion) | Summary of achievements |
| **Chapter 5** | [Future Work](#chapter-5--future-work) | Areas of improvement |
| **Chapter 6** | [References](#chapter-6--references) | Academic and technical sources |
| **Appendix** | [List of Figures](#list-of-figures) | All diagrams and screenshots |

---

## LIST OF FIGURES

| Figure | Title | Section |
|--------|-------|---------|
| Fig 1 | System Architecture Diagram | [2.2](#22-system-architecture) |
| Fig 2 | End-to-End Pipeline Flow Chart | [2.3](#23-workflow) |
| Fig 3 | Analysis Pipeline — 5-Stage Orchestration | [2.3](#23-workflow) |
| Fig 4 | Knowledge Graph Schema | [2.2](#22-system-architecture) |
| Fig 5 | Temporal Analysis Data Flow | [2.3](#23-workflow) |
| Fig 6 | Safety Engine Architecture | [2.2](#22-system-architecture) |
| Fig 7 | Project Directory Structure | [3.2](#32-initialization-and-setup) |
| Fig 8 | Phase-wise Development Timeline | [3.3](#33-implementation) |
| Fig 9 | Frontend Dashboard (Light Mode) | [3.3.6](#phase-5-symbolic-constraints--knowledge-graph) |
| Fig 10 | Frontend Dashboard (Dark Mode with Neon Effects) | [3.3.6](#phase-5-symbolic-constraints--knowledge-graph) |
| Fig 11 | API Endpoint Map | [3.3.6](#phase-5-symbolic-constraints--knowledge-graph) |

---

# Chapter 1 — Introduction

## 1.1 Introduction

**NS-TEA (Neuro-Symbolic Temporal EHR Agent)** is an advanced AI-powered Clinical Decision Support System (CDSS) that combines the generative reasoning capabilities of Large Language Models (LLMs) with the precision and safety guarantees of symbolic AI, knowledge graphs, and temporal patient analysis.

Modern healthcare generates enormous volumes of Electronic Health Records (EHR), clinical guidelines, lab results, medication histories, and temporal patient data. Clinicians face the impossible task of synthesizing all this information for every patient encounter — leading to diagnostic errors, drug interactions, missed safety checks, and delayed treatments. Studies estimate that diagnostic errors affect approximately 12 million adults annually in the United States alone (Singh et al., 2014), and preventable adverse drug events account for over $3.5 billion in excess medical costs per year (Bates et al., 1997).

NS-TEA addresses this challenge through a **hybrid neuro-symbolic architecture** that:

1. **Retrieves** relevant clinical guidelines using Retrieval-Augmented Generation (RAG) with semantic embeddings
2. **Analyzes** patient temporal history using graph-based time-decay models to identify trends and causal patterns
3. **Reasons** through structured 6-step clinical reasoning powered by a cloud-hosted 120B-parameter LLM
4. **Validates** every recommendation through a dual-layer safety engine combining YAML rule engines and a symbolic knowledge graph with formal constraint checking
5. **Gates** all outputs through a multi-dimensional confidence scoring system with automatic human-in-the-loop escalation

The system is designed to augment — not replace — clinical decision-making, providing evidence-grounded suggestions that are transparent, explainable, and verifiable.

### Key Differentiators

| Approach | Pure LLM | Pure Expert System | **NS-TEA (Hybrid)** |
|----------|----------|-------------------|----------------------|
| Reasoning Flexibility | ✅ High | ❌ Rigid | ✅ High |
| Safety Guarantees | ❌ None | ✅ Strong | ✅ Strong |
| Evidence Grounding | ❌ Hallucination risk | ✅ Hard-coded | ✅ RAG-retrieved |
| Temporal Awareness | ❌ Stateless | ⚠️ Limited | ✅ Graph-based decay |
| Explainability | ⚠️ Opaque | ✅ Full trace | ✅ Step-by-step reasoning |
| Confidence Gating | ❌ None | ❌ None | ✅ Multi-dimensional |

## 1.2 Problem Statement

Clinical decision-making in modern healthcare is plagued by several interconnected challenges:

### 1.2.1 Information Overload
Physicians must process vast amounts of patient data — lab results, medication histories, comorbidities, clinical guidelines, temporal trends — often under extreme time pressure. The average primary care physician would need to spend 26.7 hours per day to follow all recommended guidelines for a standard panel of patients (Østbye et al., 2005).

### 1.2.2 Diagnostic Error
Diagnostic errors are the most common, most costly, and most dangerous of medical errors. Approximately 5% of US adults who seek outpatient care each year experience a diagnostic error (Singh & Graber, 2015). These errors often stem from:
- **Anchoring bias**: Over-reliance on initial impressions
- **Premature closure**: Failing to consider differential diagnoses
- **Availability bias**: Overweighting recent or memorable cases
- **Information fragmentation**: Critical data scattered across systems

### 1.2.3 Drug Safety Failures
Adverse drug events (ADEs) cause over 1.3 million emergency department visits annually in the US. Many are preventable through:
- Cross-checking drug-drug interactions
- Allergy cross-reactivity screening
- Condition-specific contraindication validation
- Guideline-based dosing verification

### 1.2.4 Lack of Temporal Reasoning
Existing CDSS systems treat patient data as static snapshots, ignoring the critical temporal dimension:
- **Disease progression patterns** over weeks and months
- **Medication response trajectories** and timing
- **Lab trend analysis** (worsening kidney function, improving HbA1c)
- **Causal chains** (NSAID use → gastric bleeding → iron deficiency anemia)

### 1.2.5 LLM Limitations in Healthcare
While LLMs show promising reasoning capabilities, they suffer from:
- **Hallucination**: Generating plausible but incorrect medical information
- **No safety guarantees**: Unable to formally verify drug interactions
- **Lack of provenance**: Cannot cite specific guidelines for recommendations
- **Overconfidence**: No mechanism to flag uncertainty or request human review

**NS-TEA solves these problems** by combining the reasoning power of LLMs with the formal safety guarantees of symbolic AI, the evidence grounding of RAG, and the temporal awareness of patient graph analysis.

## 1.3 Aims and Objectives

### Primary Aim
Design and implement a production-grade Neuro-Symbolic Clinical Decision Support System that provides safe, explainable, evidence-based clinical recommendations by combining neural LLM reasoning with symbolic constraint validation.

### Objectives

| # | Objective | Phase | Status |
|---|-----------|-------|--------|
| O1 | Build a structured clinical reasoning pipeline using LLMs with validated output parsing | Phase 0 | ✅ Complete |
| O2 | Implement Retrieval-Augmented Generation for evidence-grounded guideline retrieval | Phase 1 | ✅ Complete |
| O3 | Develop a YAML-driven drug safety rule engine covering allergy, interaction, and contraindication checks | Phase 1 | ✅ Complete |
| O4 | Create a multi-dimensional confidence scoring system with automatic HITL escalation | Phase 2 | ✅ Complete |
| O5 | Design a structured 6-step clinical reasoning prompt with differential diagnosis and evidence tracking | Phase 3 | ✅ Complete |
| O6 | Build a temporal patient graph with time-decay importance weighting and causal link discovery | Phase 4 | ✅ Complete |
| O7 | Implement an embedding cache with batch pre-computation for production-scale temporal analysis | Phase 4 | ✅ Complete |
| O8 | Construct a clinical knowledge graph with symbolic constraint engine for formal safety validation | Phase 5 | ✅ Complete |
| O9 | Develop a production web interface with interactive calculators, dark mode, and real-time analysis | Phase 5 | ✅ Complete |
| O10 | Achieve comprehensive test coverage (100+ tests) across all system components | All | ✅ 104 tests |

## 1.4 Features

### Core Clinical Features

- **AI-Powered Diagnosis**: Structured 6-step clinical reasoning with primary and differential diagnosis generation, supporting and contradicting evidence tracking, probability-weighted differentials
- **Drug Safety Validation**: Dual-layer safety (YAML rules + Knowledge Graph), allergy direct match and cross-reactivity screening, drug-drug interaction detection, condition-drug contraindication checks
- **Evidence-Based Recommendations**: RAG-grounded guideline retrieval from 6 clinical corpora (ACS, Anticoagulation, Diabetes, Hypertension, Pregnancy Drug Safety, UTI Treatment)
- **Temporal Patient Analysis**: NetworkX-based patient history graphs, exponential time-decay weighting, causal link discovery (16 known relationships), graph centrality importance scoring
- **Clinical Calculators**: eGFR (CKD-EPI 2021 race-free), CHA₂DS₂-VASc (stroke risk), MELD-Na (liver disease), Wells DVT (thrombosis probability), CURB-65 (pneumonia severity)
- **Confidence Gating**: Multi-dimensional scoring (overall, evidence strength, model certainty), automatic human-in-the-loop escalation, explicit uncertainty acknowledgment

### Technical Features

- **FastAPI REST API**: 11 endpoints with Pydantic validation, CORS support, correlation ID tracking
- **Premium Web Interface**: Alpine.js + TailwindCSS SPA; dark mode with neon glow effects; 3D molecular particle background; glass morphism cards; responsive design
- **FHIR Data Ingestion**: Parses standard FHIR R4 bundles (Patient, Condition, Medication, Allergy, Observation)
- **Structured Logging**: structlog with JSON output and request correlation IDs
- **Comprehensive Testing**: 104 tests across 6 test modules covering all core components

## 1.5 Future Scope

NS-TEA is designed as an extensible platform with clear upgrade paths:

- **Phase 6 — System Hardening**: Rate limiting, input sanitization, audit logging, circuit breakers, graceful degradation
- **Phase 7 — Graph Neural Networks**: Replace rule-based temporal analysis with trained GNN models (PyTorch Geometric) for automatic pattern discovery
- **Phase 8 — Production Database**: Migrate from in-memory stores to Neo4j (knowledge graph), PostgreSQL (patient data), Redis (caching)
- **Phase 9 — Multi-Modal Inputs**: Support for medical imaging (X-ray, CT, MRI) alongside structured EHR data
- **Phase 10 — Federated Learning**: Privacy-preserving model improvement across hospital networks without centralizing patient data
- **Cloud Deployment**: Vercel/AWS/Azure deployment with horizontal scaling, monitoring dashboards, and A/B testing

---

# Chapter 2 — Project Plan

## 2.1 Introduction

NS-TEA was developed using an **iterative phased methodology**, where each phase builds upon the previous one, adding new capabilities while maintaining backward compatibility and full test coverage. This approach ensures that:

1. **Each phase is independently testable** — core functionality never regresses
2. **Safety systems are additive** — each layer adds protection without removing previous checks
3. **The system degrades gracefully** — if any component fails, the pipeline continues with reduced but safe functionality

### Development Methodology

The project follows an **incremental delivery model** with 6 completed phases:

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5
  LLM       RAG +        Confidence   Prompt       Temporal     Symbolic
  Only      Safety       Gating       Engineering  Analysis     Constraints
            Rules                                  + Graphs     + KG + UI
```

Each phase was developed with:
- **Design → Implement → Test → Evaluate** cycle
- Automated test suite run after every change
- Manual evaluation using curated clinical test cases
- Safety validation against known drug interaction scenarios

### Codebase Statistics

| Metric | Value |
|--------|-------|
| Total Python modules | 32 |
| Lines of code (Python) | ~5,500+ |
| Lines of code (Frontend) | ~1,600+ |
| Test cases | 104 |
| API endpoints | 11 |
| Clinical guidelines indexed | 6 |
| Known drug safety rules | 40+ |
| Knowledge graph nodes | 30+ |
| Knowledge graph edges | 20+ |
| Clinical calculators | 5 |

## 2.2 System Architecture

### Fig 1 — System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    NS-TEA ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐                │
│   │                     PRESENTATION LAYER                          │                │
│   │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │                │
│   │  │  Web UI       │  │  Streamlit   │  │  REST API (FastAPI)   │ │                │
│   │  │  Alpine.js +  │  │  Phase 0-1   │  │  /api/v1/*            │ │                │
│   │  │  TailwindCSS  │  │  Interface   │  │  11 endpoints         │ │                │
│   │  └──────┬───────┘  └──────────────┘  └──────────┬────────────┘ │                │
│   └─────────┼────────────────────────────────────────┼──────────────┘                │
│             │              HTTP/JSON                  │                               │
│   ┌─────────▼────────────────────────────────────────▼──────────────┐                │
│   │                     APPLICATION LAYER                           │                │
│   │                                                                 │                │
│   │  ┌─────────────────────────────────────────────────────────┐    │                │
│   │  │              ORCHESTRATOR (orchestrator.py)              │    │                │
│   │  │   ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐   │    │                │
│   │  │   │ Stage 1:  │ │ Stage 2: │ │Stage 3:│ │ Stage 4:  │   │    │                │
│   │  │   │ RAG +     │→│ Safety   │→│ LLM    │→│ Safety    │   │    │                │
│   │  │   │ Temporal  │ │ PreCheck │ │Reason  │ │ PostCheck │   │    │                │
│   │  │   └──────────┘ └──────────┘ └────────┘ └───────────┘   │    │                │
│   │  │                         │                                │    │                │
│   │  │                ┌────────▼────────┐                       │    │                │
│   │  │                │    Stage 5:     │                       │    │                │
│   │  │                │   Confidence    │                       │    │                │
│   │  │                │     Gate        │                       │    │                │
│   │  │                └─────────────────┘                       │    │                │
│   │  └─────────────────────────────────────────────────────────┘    │                │
│   │                                                                 │                │
│   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐ │                │
│   │  │ Feedback       │ │ Calculator    │ │ FHIR Data Loader      │ │                │
│   │  │ Service        │ │ Service       │ │ (Bundle Parser)       │ │                │
│   │  └───────────────┘ └───────────────┘ └───────────────────────┘ │                │
│   └─────────────────────────────────────────────────────────────────┘                │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐                │
│   │                      INTELLIGENCE LAYER                         │                │
│   │                                                                 │                │
│   │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │                │
│   │  │  RAG Engine   │  │  Temporal    │  │  LLM Reasoning Agent  │ │                │
│   │  │  ────────     │  │  Engine      │  │  ──────────────────   │ │                │
│   │  │  Embedder     │  │  ──────      │  │  Agno Framework       │ │                │
│   │  │  (MiniLM-L6)  │  │  Graph       │  │  Ollama Cloud Backend │ │                │
│   │  │  VectorStore  │  │  Builder     │  │  (gpt-oss:120b)       │ │                │
│   │  │  (Numpy)      │  │  Temporal    │  │  Structured Prompts   │ │                │
│   │  │  Context      │  │  Encoder     │  │  6-Step Reasoning     │ │                │
│   │  │  Builder      │  │  Cache       │  │  JSON Output Parsing  │ │                │
│   │  │              │  │  Batch       │  │                       │ │                │
│   │  │  6 Guideline  │  │  Updater     │  │  Retry + Timeout      │ │                │
│   │  │  Corpora      │  │             │  │  (120s LLM, 180s tot) │ │                │
│   │  └──────────────┘  └──────────────┘  └───────────────────────┘ │                │
│   └─────────────────────────────────────────────────────────────────┘                │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐                │
│   │                       SAFETY LAYER                              │                │
│   │                                                                 │                │
│   │  ┌──────────────────────────┐  ┌──────────────────────────────┐ │                │
│   │  │  YAML Rule Engine        │  │  Symbolic Constraint Engine  │ │                │
│   │  │  ─────────────────       │  │  ──────────────────────────  │ │                │
│   │  │  • Allergy matching      │  │  • Knowledge Graph (NetworkX)│ │                │
│   │  │  • Cross-reactivity      │  │  • Drug-drug interactions    │ │                │
│   │  │  • Drug-drug interaction  │  │  • Contraindication checks  │ │                │
│   │  │  • Condition-drug contra. │  │  • Allergy cross-reactivity │ │                │
│   │  │                          │  │  • Guideline alignment       │ │                │
│   │  │  40+ rules from YAML     │  │  • Unmapped action tracking  │ │                │
│   │  └──────────────────────────┘  └──────────────────────────────┘ │                │
│   └─────────────────────────────────────────────────────────────────┘                │
│                                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐                │
│   │                        DATA LAYER                               │                │
│   │  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────────┐  │                │
│   │  │ FHIR     │ │ Clinical │ │ Safety     │ │ Feedback        │  │                │
│   │  │ Bundles  │ │ Guidelines│ │ Rules YAML │ │ JSON Store      │  │                │
│   │  │ (JSON)   │ │ (6 .txt) │ │ (2 .yml)  │ │                 │  │                │
│   │  └──────────┘ └──────────┘ └────────────┘ └─────────────────┘  │                │
│   └─────────────────────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Fig 4 — Knowledge Graph Schema

```
                        ┌─────────────────┐
                        │   GUIDELINE     │
                        │   ─────────     │
                        │ • acs_guideline │
                        │ • diabetes_gl   │
                        │ • anticoag_gl   │
                        │ • htn_guideline │
                        │ • pregnancy_gl  │
                        └──────┬──────────┘
                   recommends/│\for_condition
                             │  \
            ┌────────────────▼───▼──────────────┐
            │           DRUG (30+ nodes)         │
            │  ────────────────────────          │
            │  metformin, lisinopril, aspirin,   │
            │  warfarin, heparin, amiodarone,    │
            │  metoprolol, atorvastatin,         │
            │  clopidogrel, ibuprofen, ...       │
            └─┬──────────────┬───────────────┬──┘
              │              │               │
   interacts_with    contraindicated_in    causes_allergy
   requires_caution         │            cross_reacts_with
              │              │               │
              ▼              ▼               ▼
      ┌──────────┐   ┌──────────────┐  ┌──────────────┐
      │ DRUG     │   │  CONDITION   │  │  ALLERGY     │
      │ (other)  │   │  ──────────  │  │  ────────    │
      │          │   │  CKD, CHF,   │  │  penicillin, │
      │          │   │  liver_dz,   │  │  sulfa,      │
      │          │   │  diabetes,   │  │  NSAID,      │
      │          │   │  GI_bleed,   │  │  ACE_inh,    │
      │          │   │  asthma...   │  │  statin...   │
      └──────────┘   └──────────────┘  └──────────────┘

EDGE TYPES:
  ──── interacts_with (critical)    Drug ↔ Drug     Bidirectional
  ──── requires_caution (warning)   Drug ↔ Drug     Bidirectional
  ──── contraindicated_in           Drug → Condition Directional
  ──── causes_allergy               Substance → Allergy  Directional
  ──── cross_reacts_with            Allergy → Drug  Directional
  ──── recommends                   Guideline → Drug  Directional
  ──── for_condition                Guideline → Condition  Directional
```

### Fig 6 — Safety Engine Architecture

```
┌────────────────────────────── PATIENT INPUT ─────────────────────────────────┐
│  Conditions: [diabetes, CKD_stage3]                                          │
│  Medications: [metformin, lisinopril]                                        │
│  Allergies: [penicillin_allergy]                                             │
│  Proposed Drugs: [amoxicillin, ibuprofen, aspirin]                           │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │          LAYER 1: YAML RULE ENGINE          │
              │                                            │
              │  contraindications.yml → 20+ rules         │
              │  drug_interactions.yml → 20+ rules         │
              │                                            │
              │  Check: allergy_match(penicillin→amoxicillin) │
              │  Check: cross_reactivity(penicillin→*)      │
              │  Check: condition_drug(CKD→ibuprofen)       │
              │  Check: drug_interaction(aspirin+warfarin)   │
              │                                            │
              │  Result: SafetyResult(violations[])         │
              └─────────────────────┬──────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │     LAYER 2: SYMBOLIC CONSTRAINT ENGINE      │
              │                                            │
              │  Knowledge Graph queries:                   │
              │  ├── get_drug_interactions(drug)            │
              │  ├── get_contraindications(drug)            │
              │  ├── get_allergy_cross_reactions(allergy)   │
              │  └── get_guideline_recommendations(cond)    │
              │                                            │
              │  Formal validation:                         │
              │  ├── Edge traversal for interaction chains  │
              │  ├── Guideline alignment scoring            │
              │  └── Unmapped action identification         │
              │                                            │
              │  Result: ConstraintResult(violations[])     │
              └─────────────────────┬──────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │           MERGED SAFETY OUTPUT              │
              │                                            │
              │  SafetyFlags:                              │
              │  🔴 CRITICAL: Penicillin allergy →         │
              │               Block amoxicillin            │
              │  🔴 CRITICAL: CKD Stage 3 →               │
              │               Block ibuprofen (NSAID)      │
              │  ⚠️  WARNING: Aspirin + existing meds →    │
              │               Monitor for bleeding risk    │
              │  ℹ️  INFO: Aspirin aligns with ACS         │
              │            guideline for cardiac patients   │
              │                                            │
              │  Actions: Remove blocked recommendations    │
              │           Flag for human review             │
              └────────────────────────────────────────────┘
```

## 2.3 Workflow

### Fig 2 — End-to-End Pipeline Flow Chart

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ CLINICIAN │────→│  WEB UI /    │────→│  FastAPI      │────→│ ORCHESTRATOR │
│ INPUT     │     │  REST API    │     │  Validation   │     │              │
└──────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                   │
                    ┌──────────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    STAGE 1 (PARALLEL)                         │
    │                                                               │
    │  ┌─────────────────────┐    ┌──────────────────────────────┐ │
    │  │    RAG RETRIEVAL    │    │    TEMPORAL ANALYSIS          │ │
    │  │                     │    │                               │ │
    │  │ Patient Summary     │    │ Build Patient Graph           │ │
    │  │       ↓             │    │       ↓                       │ │
    │  │ Embed Query         │    │ Compute Time-Decay Weights    │ │
    │  │       ↓             │    │       ↓                       │ │
    │  │ Search VectorStore  │    │ Calculate Centrality          │ │
    │  │       ↓             │    │       ↓                       │ │
    │  │ Top-5 Guidelines    │    │ Generate Temporal Insights    │ │
    │  └──────────┬──────────┘    └──────────────┬───────────────┘ │
    │             │              Merged Context    │                 │
    └─────────────┼───────────────────────────────┼─────────────────┘
                  └───────────────┬───────────────┘
                                  ▼
    ┌────────────────────────────────────────────────────────────────┐
    │  STAGE 2: SAFETY PRE-CHECK                                     │
    │  ┌─────────────────────┐   ┌──────────────────────────────┐   │
    │  │  YAML Rule Engine   │ + │  Symbolic Constraint Engine  │   │
    │  │  (current meds)     │   │  (knowledge graph queries)   │   │
    │  └─────────────────────┘   └──────────────────────────────┘   │
    │  Output: Pre-existing safety flags for LLM context             │
    └───────────────────────────────┬────────────────────────────────┘
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │  STAGE 3: LLM REASONING                                       │
    │                                                                │
    │  Input: Patient Summary + RAG Context + Temporal Insights      │
    │         + Safety Pre-Check Results                             │
    │                                                                │
    │  LLM: gpt-oss:120b-cloud (via Agno + Ollama)                 │
    │  Timeout: 120 seconds per attempt                              │
    │  Retries: Up to 2 attempts                                    │
    │                                                                │
    │  6-Step Structured Reasoning:                                  │
    │    1. Patient Assessment — Synthesize key findings              │
    │    2. Evidence Retrieval — Map findings to guidelines           │
    │    3. Differential Diagnosis — Rank by probability              │
    │    4. Risk Assessment — Safety and temporal factors             │
    │    5. Treatment Planning — Evidence-based recommendations       │
    │    6. Monitoring Plan — Follow-up and reassessment              │
    │                                                                │
    │  Output: Structured JSON (diagnosis, differential,             │
    │          recommendations, reasoning steps, safety flags)        │
    └───────────────────────────────┬────────────────────────────────┘
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │  STAGE 4: SAFETY POST-CHECK                                    │
    │                                                                │
    │  Extract proposed drugs from LLM recommendations               │
    │  ┌─────────────────────┐   ┌──────────────────────────────┐   │
    │  │  YAML Rule Engine   │ + │  Symbolic Constraint Engine  │   │
    │  │  (proposed drugs)   │   │  (formal graph validation)   │   │
    │  └─────────────────────┘   └──────────────────────────────┘   │
    │                                                                │
    │  Actions:                                                      │
    │  • Convert violations → SafetyFlags                            │
    │  • REMOVE recommendations with critical violations             │
    │  • Add guideline alignment info flags                          │
    │  • Track unmapped actions for transparency                     │
    └───────────────────────────────┬────────────────────────────────┘
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │  STAGE 5: CONFIDENCE GATE                                      │
    │                                                                │
    │  Multi-dimensional scoring:                                    │
    │    ├── Overall confidence < 0.65 → escalate                   │
    │    ├── < 2 reasoning steps → escalate                         │
    │    ├── 0 recommendations → escalate                           │
    │    ├── Critical safety flags → escalate                       │
    │    ├── Fallback diagnosis detected → escalate                 │
    │    └── Evidence strength < 0.4 → escalate                    │
    │                                                                │
    │  Output: requires_human_review (bool)                          │
    │          escalation_reason (string)                             │
    └───────────────────────────────┬────────────────────────────────┘
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                     ANALYSIS RESPONSE                          │
    │                                                                │
    │  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
    │  │Diagnosis │ │Differential│ │Recs   │ │Safety │ │Confid- │  │
    │  │(primary) │ │(ranked)   │ │(safe) │ │Flags  │ │ence    │  │
    │  └──────────┘ └──────────┘ └────────┘ └────────┘ └────────┘  │
    │  ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐     │
    │  │Reasoning     │ │Human Review    │ │Raw LLM Output    │     │
    │  │Steps (trace) │ │Decision        │ │(for audit)       │     │
    │  └──────────────┘ └────────────────┘ └──────────────────┘     │
    └───────────────────────────────┬────────────────────────────────┘
                                    ▼
    ┌────────────────────────────────────────────────────────────────┐
    │  CLINICIAN REVIEW                                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
    │  │ ✅ ACCEPT │  │ ✏️ MODIFY │  │ ❌ REJECT │                     │
    │  └──────────┘  └──────────┘  └──────────┘                     │
    │            Feedback stored → Continuous improvement             │
    └────────────────────────────────────────────────────────────────┘
```

### Fig 3 — Analysis Pipeline (5-Stage Orchestration)

```
              Patient Input
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             │
 ┌──────┐   ┌─────────┐        │
 │ RAG  │   │Temporal │    (parallel)
 │Search│   │ Graph   │        │
 └──┬───┘   └────┬────┘        │
    │             │             │
    └──────┬──────┘             │
           ▼                    │
    ┌──────────────┐            │
    │ Safety       │ ◄──────────┘
    │ Pre-Check    │     (current meds)
    └──────┬───────┘
           │  context injected
           ▼
    ┌──────────────┐
    │   LLM        │ ←── 120s timeout
    │ Reasoning    │ ←── 2 retries
    └──────┬───────┘
           │  proposed recommendations
           ▼
    ┌──────────────┐
    │ Safety       │
    │ Post-Check   │ ←── blocks unsafe drugs
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Confidence   │
    │ Gate         │ ←── HITL escalation
    └──────┬───────┘
           │
           ▼
    AnalysisResponse
```

### Fig 5 — Temporal Analysis Data Flow

```
Patient History Events            Temporal Graph (NetworkX DiGraph)
──────────────────────            ───────────────────────────────
                                  
 2024-01  Diabetes diagnosed ──→  [diabetes_dx] ──→ [metformin_start]
 2024-03  Metformin started  ──→       │                    │
 2024-06  HbA1c = 8.1%      ──→  [hba1c_8.1] ──────────────┘
 2024-09  Hypertension dx    ──→  [htn_dx] ──→ [lisinopril_start]
 2024-10  Lisinopril started ──→       │
 2025-01  Creatinine = 1.8  ──→  [creatinine_1.8] (ABNORMAL 1.5x)
 2025-03  Chest pain visit   ──→  [chest_pain_visit]
                                       │
                                  Known Causal Links Applied:
                                  diabetes ──→ cardiovascular risk
                                  diabetes ──→ renal risk
                                  hypertension ──→ cardiovascular risk
                                  hypertension ──→ stroke risk
                                  
                                  Time-Decay Weighting:
                                  ┌─────────────────────────────────┐
                                  │ importance = e^(-0.005 × days)  │
                                  │                                 │
                                  │ chest_pain (30d ago)  = 0.86    │
                                  │ creatinine (90d ago)  = 0.64    │
                                  │ htn_dx (180d ago)     = 0.41    │
                                  │ diabetes_dx (450d ago)= 0.11    │
                                  └─────────────────────────────────┘
                                  
                                  Top Insights Generated:
                                  1. "Recent chest pain (30 days) — HIGH priority"
                                  2. "Abnormal creatinine 1.8 — monitor renal function"
                                  3. "Diabetes + Hypertension → elevated cardiovascular risk"
```

## 2.4 Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.12.10 | Primary development language |
| **Git** | Latest | Version control |
| **VS Code** | Latest | Primary IDE |
| **GitHub Copilot** | Latest | AI pair programming assistant |
| **Ollama Cloud** | Latest | LLM hosting (gpt-oss:120b-cloud model) |
| **pytest** | 8.0+ | Test framework |
| **ruff** | 0.4+ | Linting and formatting |
| **mypy** | 1.10+ | Static type checking |
| **PowerShell** | 5.1 | Terminal and scripting (Windows) |
| **Invoke-RestMethod** | (PowerShell) | API testing |

## 2.5 Platform Used in Development

| Component | Details |
|-----------|---------|
| **Operating System** | Windows 10/11 |
| **Python Runtime** | CPython 3.12.10 (virtual environment: `.final_proj`) |
| **IDE** | Visual Studio Code with Python, Pylance, GitHub Copilot extensions |
| **Package Manager** | pip (with hatchling build backend) |
| **Virtual Environment** | Python venv (`D:\AI\Exploration\.final_proj`) |
| **Browser** | Microsoft Edge / Chromium (for UI testing) |
| **LLM Backend** | Ollama Cloud (remote API), model: `gpt-oss:120b-cloud` |
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers, runs locally via CPU) |

## 2.6 Libraries Used

### Core Framework

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **FastAPI** | ≥0.100 | Web framework | High-performance async REST API with automatic OpenAPI documentation, Pydantic integration, dependency injection. Handles all 11 endpoints including analysis, calculators, feedback, and static file serving. |
| **Uvicorn** | ≥0.20 | ASGI server | Lightning-fast ASGI server for serving FastAPI. Supports auto-reload in development, configurable workers for production. |
| **Pydantic** | ≥2.0 | Data validation | Type-safe data models for all inputs (PatientInput, FeedbackInput) and outputs (AnalysisResponse, ConfidenceScore). Provides automatic validation, serialization, and JSON Schema generation. |
| **pydantic-settings** | ≥2.0 | Configuration | Environment-based settings management. Reads from `.env` files with type validation for model_provider, model_id, temperature, max_tokens, paths. |

### AI / Machine Learning

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **Agno** | ≥2.5 | Agent framework | Provides the `Agent` class for structured LLM interactions. Supports Ollama and HuggingFace backends, tool integration, and conversation management. Used in reasoning agents (Phase 0-3). |
| **sentence-transformers** | ≥2.0 | Embedding model | Provides the `all-MiniLM-L6-v2` model for converting clinical text into 384-dimensional dense vectors. Used for RAG query encoding and guideline chunk embedding. CPU-optimized for local inference. |
| **huggingface_hub** | ≥0.20 | Model hub access | Downloads and caches sentence-transformer models. Provides API tokens for authenticated model access. |

### Data & Graph

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **NetworkX** | ≥3.0 | Graph library | Powers both the Temporal Patient Graph (Phase 4) and Clinical Knowledge Graph (Phase 5). Supports directed graphs (DiGraph), node/edge attributes, degree centrality calculation, neighbor traversal, and graph serialization. |
| **pandas** | ≥2.0 | Data processing | Tabular data manipulation for FHIR bundle parsing, lab result aggregation, evaluation metrics computation. Used in data loading and evaluation scripts. |
| **orjson** | ≥3.9 | Fast JSON | High-performance JSON serialization/deserialization. 3-10x faster than stdlib `json`. Used for FHIR bundle parsing, feedback storage, and API response generation. |

### Logging & Infrastructure

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **structlog** | ≥24.0 | Structured logging | JSON-formatted structured logging with context variables. Supports correlation ID binding for request tracing across the pipeline. Integrates with Python's `logging` module. |
| **python-dotenv** | ≥1.0 | Environment management | Loads environment variables from `.env` files. Manages secrets like `NSTEA_OLLAMA_API_KEY` without hardcoding. |

### Frontend (Server-side)

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **Streamlit** | ≥1.30 | Phase 0-1 UI | Rapid prototyping UI framework. Provides form inputs, result display, and interactive analysis for early development phases. |

### Frontend (Client-side CDN)

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **Alpine.js** | 3.x | Reactive UI framework | Lightweight reactive JavaScript framework (15KB). Powers all UI interactivity: page navigation, form state management, API calls, dark mode toggle. Loaded via CDN. |
| **TailwindCSS** | 3.x | CSS framework | Utility-first CSS framework. Provides responsive design, dark mode (`darkMode: 'class'`), custom color palettes (neon, clinical, surface), glassmorphism, and animation classes. Loaded via CDN. |
| **Chart.js** | 4.x | Data visualization | Canvas-based charting library. Renders doughnut charts for feedback distribution (accept/modify/reject). Lightweight and responsive. |
| **Google Fonts** | — | Typography | Inter (UI text, weights 300-900) and JetBrains Mono (code/monospace) fonts for professional medical appearance. |

### Development & Testing

| Library | Version | Purpose | Details |
|---------|---------|---------|---------|
| **pytest** | ≥8.0 | Test framework | Primary testing framework. 104 test cases across 6 modules. Supports parametrized tests, fixtures, and assertion introspection. |
| **pytest-asyncio** | ≥0.23 | Async testing | Enables `async def test_*` functions. Required for testing async pipeline stages, LLM calls, and FastAPI endpoints. Auto mode configured in `pytest.ini`. |
| **ruff** | ≥0.4 | Linter/formatter | Extremely fast Python linter (written in Rust). Enforces style rules: E (pycodestyle), F (pyflakes), I (isort), W (warnings). |
| **mypy** | ≥1.10 | Type checker | Static type analysis. Validates Pydantic models, function signatures, and return types across the codebase. |

---

# Chapter 3 — Working of Project

## 3.1 Research and Requirements Gathering

### Literature Review

The design of NS-TEA was informed by research in the following areas:

1. **Clinical Decision Support Systems (CDSS)**: Review of existing systems like IBM Watson Health, Epic CDS, and UpToDate — identifying limitations in transparency, temporal reasoning, and LLM grounding.

2. **Retrieval-Augmented Generation (RAG)**: Based on Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — establishing the pattern of retrieve-then-generate for factual grounding.

3. **Neuro-Symbolic AI**: Inspired by Garcez et al. (2019) and Marcus (2020) on combining neural networks with symbolic reasoning for safety-critical applications.

4. **Temporal Reasoning in Healthcare**: Based on research into temporal phenotyping and disease trajectory modeling (Jensen et al., 2014; Rajkomar et al., 2018).

5. **Drug Safety Ontologies**: Leveraged established drug interaction databases and classification systems (DrugBank, RxNorm, ICD-10, SNOMED CT, LOINC).

6. **Clinical Scoring Systems**: Implemented validated medical calculators following original publications (CKD-EPI 2021, CHA₂DS₂-VASc, MELD-Na, Wells DVT, CURB-65).

### Requirements Specification

| ID | Requirement | Priority | Type |
|----|-------------|----------|------|
| R1 | Accept structured patient input (demographics, conditions, medications, allergies, labs, vitals, history) | Must | Functional |
| R2 | Generate primary diagnosis with differential ranking | Must | Functional |
| R3 | Provide evidence-based treatment recommendations | Must | Functional |
| R4 | Block unsafe drug recommendations (allergy, interaction, contraindication) | Must | Safety |
| R5 | Ground reasoning in clinical guidelines via RAG | Must | Quality |
| R6 | Analyze temporal patient history for trends and causal patterns | Should | Functional |
| R7 | Score confidence multi-dimensionally and escalate low-confidence results | Must | Safety |
| R8 | Provide step-by-step reasoning trace for transparency | Must | Explainability |
| R9 | Collect clinician feedback for continuous improvement | Should | Functional |
| R10 | Expose REST API for integration with existing EHR systems | Must | Technical |
| R11 | Provide an interactive web interface for standalone use | Should | Technical |
| R12 | Validate with 100+ automated tests | Must | Quality |

## 3.2 Initialization and Setup

### Environment Setup

```powershell
# Create virtual environment
python -m venv .final_proj

# Activate
.\.final_proj\Scripts\Activate.ps1

# Install project in editable mode
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with:
#   NSTEA_MODEL_PROVIDER=ollama
#   NSTEA_MODEL_ID=gpt-oss:120b-cloud
#   NSTEA_OLLAMA_HOST=https://your-ollama-host
#   NSTEA_OLLAMA_API_KEY=your-api-key

# Index guidelines for RAG
python scripts/index_guidelines.py

# Run tests
pytest -v

# Start server
python -m uvicorn nstea.api:app --host 0.0.0.0 --port 8001 --reload
```

### Fig 7 — Project Directory Structure

```
D:\AI\Exploration\
│
├── src/nstea/                      # Main application package
│   ├── __init__.py
│   ├── main.py                     # Entry point (uvicorn runner)
│   ├── config.py                   # Pydantic settings (env-based)
│   │
│   ├── api/                        # REST API layer
│   │   ├── __init__.py             # FastAPI app factory (v0.5.0)
│   │   ├── middleware.py           # Correlation ID middleware
│   │   └── routes/
│   │       ├── analysis.py         # POST /analyze, /analyze/quick
│   │       ├── calculators.py      # 5 clinical calculator endpoints
│   │       ├── feedback.py         # Feedback CRUD endpoints
│   │       └── health.py           # GET /health
│   │
│   ├── models/                     # Pydantic data models
│   │   ├── patient.py              # PatientInput + sub-models
│   │   ├── analysis.py             # AnalysisResponse + sub-models
│   │   └── feedback.py             # FeedbackInput, FeedbackRecord
│   │
│   ├── agents/                     # LLM reasoning + orchestration
│   │   ├── orchestrator.py         # Phase 5 main pipeline
│   │   ├── reasoning_agent.py      # Phase 0 LLM-only agent
│   │   ├── reasoning_agent_v1.py   # Phase 1-3 RAG-grounded agent
│   │   ├── confidence_agent.py     # Confidence gating + HITL
│   │   ├── safety_agent.py         # Post-hoc safety validation
│   │   ├── temporal_agent.py       # Phase 4 temporal integration
│   │   └── prompts/
│   │       ├── reasoning.py        # v0.1 prompts
│   │       └── reasoning_v1.py     # v3 structured prompts
│   │
│   ├── safety/                     # Dual-layer safety engine
│   │   ├── rule_engine.py          # YAML-driven rule engine
│   │   ├── constraint_engine.py    # Phase 5 symbolic engine
│   │   └── knowledge_graph.py      # Phase 5 clinical KG
│   │
│   ├── temporal/                   # Phase 4 temporal analysis
│   │   ├── graph_builder.py        # Patient event → NetworkX graph
│   │   ├── temporal_encoder.py     # Time-decay + centrality scoring
│   │   ├── embedding_cache.py      # TTL-based embedding cache
│   │   └── batch_updater.py        # Batch pre-computation
│   │
│   ├── retrieval/                  # RAG retrieval pipeline
│   │   ├── __init__.py             # Document chunking utilities
│   │   ├── embedder.py             # Sentence-transformer wrapper
│   │   ├── vector_store.py         # Numpy-based vector search
│   │   └── context_builder.py      # Context merging for LLM
│   │
│   ├── tools/                      # Callable agent tools
│   │   ├── drug_safety.py          # Drug safety check tool
│   │   ├── guideline_search.py     # RAG search tool
│   │   └── lab_calculator.py       # 5 clinical calculators
│   │
│   ├── services/                   # Business logic services
│   │   └── feedback_service.py     # Feedback persistence
│   │
│   ├── data_layer/                 # Data ingestion
│   │   └── fhir_loader.py          # FHIR R4 bundle parser
│   │
│   └── core/                       # Infrastructure
│       └── logging.py              # structlog configuration
│
├── tests/                          # Test suite (104 tests)
│   ├── test_agents.py              # Safety + confidence tests
│   ├── test_calculators.py         # Calculator accuracy tests
│   ├── test_constraint_engine.py   # KG + constraint tests
│   ├── test_retrieval.py           # RAG pipeline tests
│   ├── test_rule_engine.py         # YAML rule tests
│   └── test_temporal.py            # Temporal analysis tests
│
├── frontend/                       # User interfaces
│   ├── web/
│   │   ├── index.html              # Production SPA (Alpine.js)
│   │   └── app.js                  # Frontend logic (~280 lines)
│   └── streamlit_app/
│       └── app.py                  # Legacy Streamlit UI
│
├── data/                           # Data assets
│   ├── guidelines/                 # RAG corpus (6 clinical texts)
│   │   ├── acs_management.txt
│   │   ├── anticoagulation_management.txt
│   │   ├── diabetes_management.txt
│   │   ├── hypertension_management.txt
│   │   ├── pregnancy_drug_safety.txt
│   │   └── uti_treatment.txt
│   ├── rules/                      # Safety rules
│   │   ├── contraindications.yml
│   │   └── drug_interactions.yml
│   ├── fhir/                       # Synthetic FHIR bundles
│   ├── test_cases/                 # Evaluation scenarios
│   ├── vector_index.pkl            # Pre-built RAG index
│   └── feedback.json               # Feedback store
│
├── scripts/                        # Utility scripts
│   ├── index_guidelines.py         # Build vector index
│   ├── run_evaluation.py           # Phase 0 eval
│   ├── run_evaluation_phase1.py    # Phase 1 eval
│   └── run_evaluation_v2.py        # Phase 2+ eval
│
├── pyproject.toml                  # Project metadata + deps
├── pytest.ini                      # Test configuration
├── .env / .env.example             # Environment config
└── README.md                       # This document
```

## 3.3 Implementation

### Fig 8 — Phase-wise Development Timeline

```
Phase 0                Phase 1               Phase 2              Phase 3
LLM Foundation         RAG + Safety          Confidence           Prompt
────────────────       ─────────────         ──────────           ──────────
│ PatientInput  │      │ Embedder     │      │ Confidence  │      │ 6-step    │
│ AnalysisResp  │      │ VectorStore  │      │ Agent       │      │ reasoning │
│ Reasoning     │  ──→ │ Context      │ ──→  │ HITL gate   │ ──→  │ v3 prompt │
│ Agent v0      │      │ Builder      │      │ Multi-dim   │      │ Evidence  │
│ Basic prompts │      │ RuleEngine   │      │ scoring     │      │ tracking  │
│ JSON parsing  │      │ YAML rules   │      │             │      │           │
└───────────────┘      │ Safety agent │      └─────────────┘      └───────────┘
                       └──────────────┘

Phase 4                          Phase 5
Temporal Analysis                Symbolic + KG + UI
──────────────────               ────────────────────────
│ Graph Builder     │            │ Knowledge Graph     │
│ Temporal Encoder  │            │ Constraint Engine   │
│ Embedding Cache   │   ──→     │ Orchestrator v5     │
│ Batch Updater     │            │ FastAPI v0.5.0      │
│ Temporal Agent    │            │ Premium Web UI      │
│ Causal Links (16) │            │ 5 Calculators       │
└───────────────────┘            │ 104 tests passing   │
                                 └─────────────────────┘
```

---

### Phase 0 — LLM Foundation

**Goal**: Establish the core data models and LLM reasoning pipeline.

**Components Built**:
- `models/patient.py` — `PatientInput` with nested Pydantic models: `Condition` (ICD-10/SNOMED codes, status), `Medication` (RxNorm codes, dosage, dates), `Allergy` (substance, reaction, severity), `Symptom`, `LabResult` (LOINC codes, reference ranges), `Vitals`, `ClinicalEvent`
- `models/analysis.py` — `AnalysisResponse` with `DiagnosisOutput`, `DifferentialDx` (probability + evidence), `Recommendation` (category, urgency, rationale), `ReasoningStep`, `SafetyFlag`, `ConfidenceScore`
- `agents/reasoning_agent.py` — LLM-only reasoning using `Agno` agent framework with Ollama backend
- `agents/prompts/reasoning.py` — System prompt establishing the clinical reasoning role and JSON output schema
- `data_layer/fhir_loader.py` — FHIR R4 bundle parser for automated patient data loading
- `scripts/run_evaluation.py` — Phase 0 evaluation harness with curated test cases

**Key Design Decision**: All input/output schemas use Pydantic v2 with strict validation. The `PatientInput.to_clinical_summary()` method generates a formatted text representation consumed by all downstream components.

---

### Phase 1 — RAG + Safety Rules

**Goal**: Ground LLM reasoning in clinical guidelines and add safety validation.

**Components Built**:
- `retrieval/embedder.py` — Wraps `sentence-transformers` (`all-MiniLM-L6-v2`), producing 384-dimensional dense vectors from clinical text
- `retrieval/vector_store.py` — Numpy-based in-memory vector store with cosine similarity search (threshold: 0.25, top-K: 5)
- `retrieval/__init__.py` — Document chunking (512 tokens, 64 overlap) for guideline corpus
- `retrieval/context_builder.py` — Merges patient summary + top-K guideline excerpts (max 4000 chars) into LLM context
- `safety/rule_engine.py` — YAML-driven safety rule engine with 4 check types:
  - **Allergy match**: Direct substance → drug matching
  - **Cross-reactivity**: Allergy to class → related drugs (e.g., penicillin allergy → block amoxicillin, ampicillin)
  - **Condition-drug contraindication**: Disease → contraindicated drugs (e.g., CKD → block NSAIDs)
  - **Drug-drug interaction**: Current medications × proposed drugs (e.g., warfarin + aspirin → bleeding risk)
- `data/rules/contraindications.yml` — Condition-drug and allergy rules
- `data/rules/drug_interactions.yml` — Drug-drug interaction rules
- `agents/reasoning_agent_v1.py` — RAG-grounded reasoning agent (pre-computed context injection)
- `agents/safety_agent.py` — Post-hoc safety validation of LLM recommendations
- `tools/drug_safety.py` — Callable tool wrapper for rule engine
- `tools/guideline_search.py` — Callable tool wrapper for RAG search (singleton vector store)
- `data/guidelines/` — 6 clinical guideline text files (ACS, anticoagulation, diabetes, hypertension, pregnancy drug safety, UTI treatment)

**Guideline Corpus**:

| File | Clinical Domain | Used For |
|------|----------------|----------|
| `acs_management.txt` | Acute Coronary Syndrome | Chest pain, MI, antiplatelet therapy |
| `anticoagulation_management.txt` | Anticoagulation | DVT, PE, AF, warfarin/heparin management |
| `diabetes_management.txt` | Diabetes Mellitus | Glucose management, insulin, metformin |
| `hypertension_management.txt` | Hypertension | BP targets, ACE-i, ARBs, diuretics |
| `pregnancy_drug_safety.txt` | Pregnancy Drug Safety | Teratogenicity, FDA categories |
| `uti_treatment.txt` | Urinary Tract Infection | Antibiotic selection, resistance |

---

### Phase 2 — Confidence Gating

**Goal**: Score output quality multi-dimensionally and automatically escalate uncertain results for human review.

**Components Built**:
- `agents/confidence_agent.py` — `evaluate_confidence(analysis)` function with 6 escalation triggers:

| Trigger | Threshold | Rationale |
|---------|-----------|-----------|
| Low overall confidence | < 0.65 | LLM reports low certainty |
| Few reasoning steps | < 2 steps | Shallow analysis indicates skip-ahead |
| No recommendations | 0 recs | Analysis failed to produce actionable output |
| Critical safety flags | Any critical | Unsafe recommendations require human review |
| Fallback diagnosis | Pattern match | Detects generic "unable to determine" responses |
| Low evidence strength | < 0.4 | Insufficient guideline support for conclusions |

**Output**: Updated `requires_human_review` (boolean) and `escalation_reason` (human-readable explanation).

---

### Phase 3 — Prompt Engineering

**Goal**: Enhance LLM reasoning quality through structured 6-step clinical reasoning prompts.

**Components Built**:
- `agents/prompts/reasoning_v1.py` — `SYSTEM_PROMPT_V1` and `REASONING_PROMPT_V3` with:

**6-Step Structured Reasoning Protocol**:
1. **Patient Assessment** — Synthesize demographics, comorbidities, medication profile, and presenting complaint
2. **Evidence Retrieval** — Map key findings to relevant clinical guidelines (from RAG context)
3. **Differential Diagnosis** — Generate ranked differential with probabilities, supporting and contradicting evidence per diagnosis
4. **Risk Assessment** — Evaluate safety risks including drug interactions, allergies, temporal trends, and comorbidity interactions
5. **Treatment Planning** — Propose evidence-based recommendations with urgency levels (stat/urgent/routine), categories (medication/test/procedure/referral/monitoring), and guideline citations
6. **Monitoring Plan** — Define follow-up timeline, reassessment criteria, and escalation triggers

**Output Schema**: Strict JSON format with `diagnosis`, `differential_diagnosis`, `recommendations`, `reasoning_steps`, `safety_flags`, `confidence`, enforced through prompt engineering and fallback parsing.

---

### Phase 4 — Temporal Analysis

**Goal**: Build a graph-based temporal reasoning layer that models patient history as a directed graph with time-decay importance weighting.

**Components Built**:
- `temporal/graph_builder.py` — `PatientGraphBuilder`:
  - Converts `ClinicalEvent` list → NetworkX `DiGraph` with temporal edges
  - Incorporates active conditions, current medications, and recent lab results as nodes
  - Applies 16 known causal links (e.g., NSAID → GI bleed, diabetes → renal risk, HF → arrhythmia)
  - Adds implicit edges between events and related diagnoses

- `temporal/temporal_encoder.py` — `TemporalEncoder`:
  - **Time-decay weighting**: `importance = e^(-λ × days_ago)` where λ=0.005 (half-life ≈ 139 days)
  - **Type importance scores**: diagnosis (1.0) > procedure (0.9) > medication (0.8) > lab (0.7) > visit (0.5) > imaging (0.8)
  - **Abnormal lab multiplier**: 1.5× for flagged results
  - **Graph centrality**: NetworkX degree centrality normalized across all nodes
  - **Combined score**: Weighted combination of all three signals
  - `get_top_insights(graph, top_k=5)` → Human-readable temporal insights for LLM context

- `temporal/embedding_cache.py` — `EmbeddingCache`:
  - In-memory dictionary with TTL (default: 86,400 seconds = 24 hours)
  - `compute_key(patient_id, patient_data)` → SHA-256 hash for change detection
  - Cache statistics: hits, misses, hit_rate
  - `cleanup_expired()` → Garbage collection of stale entries

- `temporal/batch_updater.py` — `BatchEmbeddingUpdater`:
  - Pre-computes temporal embeddings for a batch of patients
  - Combines: cache lookup → graph building → encoding → cache storage
  - Returns `BatchResult` with total, updated, cache_hits, errors, elapsed_seconds

- `agents/temporal_agent.py` — `run_temporal_analysis(patient)`:
  - Integration point between temporal layer and orchestrator
  - Builds graph → encodes → generates insights → returns `TemporalResult`
  - `TemporalResult.to_context_string()` → Formatted text injected into LLM prompt

---

### Phase 5 — Symbolic Constraints + Knowledge Graph

**Goal**: Add formal symbolic safety guarantees via a clinical knowledge graph and deploy a production-grade web interface.

**Components Built**:

**Knowledge Graph** — `safety/knowledge_graph.py`:
- `ClinicalKnowledgeGraph` backed by NetworkX
- **30+ nodes**: drugs (metformin, lisinopril, warfarin, aspirin, amiodarone, etc.), conditions (CKD, CHF, liver disease, diabetes, GI bleed, asthma, etc.), allergies (penicillin, sulfa, NSAID, ACE inhibitor, statin), guidelines (ACS, diabetes, anticoagulation, hypertension, pregnancy)
- **20+ edges**: `interacts_with`, `contraindicated_in`, `causes_allergy`, `cross_reacts_with`, `recommends`, `for_condition`, `requires_caution`
- Fuzzy name matching: `find_node(name)` strips suffixes and tries partial matches
- `build_default_knowledge_graph()` → Factory function for populated instance

**Symbolic Constraint Engine** — `safety/constraint_engine.py`:
- `SymbolicConstraintEngine` wraps the knowledge graph
- `validate(proposed_drugs, patient)` → `ConstraintResult` with:
  - Drug-drug interaction checking via graph edge traversal
  - Condition-drug contraindication validation
  - Allergy cross-reactivity detection
  - Guideline alignment scoring (are proposed drugs recommended by guidelines?)
  - Unmapped action tracking (honest incompleteness — flags drugs not in the KG)
- `requires_human_review` property: True if any unmapped actions or critical violations

**Orchestrator v5** — `agents/orchestrator.py`:
- 5-stage pipeline: RAG+Temporal (parallel) → Safety PreCheck → LLM → Safety PostCheck → Confidence Gate
- Safety PreCheck now runs **both** YAML rules + Symbolic constraints on current medications
- Safety PostCheck runs **both** engines on LLM-proposed medications
- Constraint violations merged into SafetyFlags
- Guideline alignment info merged as `info`-level safety flags
- Timeout: 120s per LLM call, 180s total pipeline

**Production Web Interface** — `frontend/web/`:
- **index.html**: Single-page application with Alpine.js + TailwindCSS
  - Dark mode toggle (persisted in localStorage)
  - 3D molecular particle background (Canvas API with mouse interaction)
  - Glass morphism cards with neon glow effects
  - 5 screens: Dashboard, New Analysis, Results, Calculators, System Health
  - Card hover-lift animations, heartbeat scan line, fade/slide-up transitions
  - Responsive design (mobile → desktop)
- **app.js**: Frontend logic
  - `nstea()` Alpine component with page routing, form state, API calls
  - `calcAPI(name, body)` → Generic calculator API wrapper
  - `loadTemplate(name)` → Pre-fill form with cardiac/diabetes/respiratory/safety templates
  - `submitFeedback()` → Clinician verdict (accept/modify/reject) with notes
  - Chart.js doughnut chart for feedback visualization

**Clinical Calculators API** — `api/routes/calculators.py`:
- 5 POST endpoints with Pydantic request validation:

| Endpoint | Calculator | Formula | Output |
|----------|-----------|---------|--------|
| `/api/v1/calculators/egfr` | eGFR (CKD-EPI 2021) | Race-free, sex-specific | mL/min/1.73m², G1-G5 staging |
| `/api/v1/calculators/cha2ds2vasc` | CHA₂DS₂-VASc | Point-based (0-9) | Stroke risk stratification |
| `/api/v1/calculators/meld` | MELD-Na | Log-based (6-40) | Liver disease severity |
| `/api/v1/calculators/wells` | Wells DVT | Point-based (-2 to 9) | DVT probability |
| `/api/v1/calculators/curb65` | CURB-65 | Additive (0-5) | Pneumonia mortality risk |

**Test Suite**: 104 tests passing across all 6 test modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_agents.py` | ~6 | Safety agent, confidence gating |
| `test_calculators.py` | ~24 | All 5 calculators, edge cases |
| `test_constraint_engine.py` | ~12 | KG queries, constraint validation |
| `test_retrieval.py` | ~15 | Embedder, vector store, RAG |
| `test_rule_engine.py` | ~18 | All 4 rule types, YAML parsing |
| `test_temporal.py` | ~29 | Graph, encoder, cache, batch |
| **Total** | **104** | **All core components** |

---

### Fig 11 — API Endpoint Map

```
                          ┌──────────────────────────────────────┐
                          │         NS-TEA API v0.5.0            │
                          │    http://localhost:8001              │
                          └──────────────────┬───────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
         Static Files                   Health Check                   API v1
              │                              │                              │
    ┌─────────┼─────────┐                    │                ┌─────────────┼────────────┐
    │         │         │                    │                │             │            │
  GET /    GET /app.js  GET /static/*     GET /health      Analysis     Feedback    Calculators
    │         │                              │                │             │            │
 index.html  app.js                    {status,          POST /analyze  POST /feedback  POST /calculators/egfr
                                        version,          POST /analyze  GET /feedback   POST /calculators/cha2ds2vasc
                                        phase}              /quick        /{id}          POST /calculators/meld
                                                                        GET /feedback   POST /calculators/wells
                                                                          -summary      POST /calculators/curb65
```

---

# Chapter 4 — Conclusion

## Summary of Achievements

NS-TEA successfully demonstrates that **neuro-symbolic hybrid architectures** can address the fundamental limitations of both pure LLM and pure expert system approaches to clinical decision support.

### Technical Achievements

1. **Hybrid Architecture**: Successfully combined neural LLM reasoning (120B-parameter model) with symbolic constraint validation (knowledge graph + rule engine), achieving both reasoning flexibility and safety guarantees in a single system.

2. **Evidence Grounding**: Implemented a complete RAG pipeline with semantic embeddings (`all-MiniLM-L6-v2`), numpy-based vector search, and context-aware merging — grounding every recommendation in indexed clinical guidelines rather than relying on LLM memory alone.

3. **Temporal Reasoning**: Built a novel graph-based temporal analysis layer that models patient history as a directed graph with exponential time-decay weighting, type-based importance scoring, and automatic causal link discovery across 16 known clinical relationships.

4. **Dual-Layer Safety**: Engineered a two-layer safety architecture where:
   - **Layer 1 (YAML rules)** provides explicit, auditable, human-authored safety rules covering allergy matching, cross-reactivity, drug-drug interactions, and condition-drug contraindications
   - **Layer 2 (Symbolic KG)** provides formal graph-based constraint validation with guideline alignment scoring and honest incompleteness tracking

5. **Confidence Gating**: Implemented multi-dimensional confidence scoring that evaluates 6 independent quality signals and automatically escalates uncertain results for human review — ensuring the system never silently produces unreliable output.

6. **Production-Grade Infrastructure**: Delivered a complete production system with:
   - 11 REST API endpoints with Pydantic validation and CORS support
   - Structured JSON logging with correlation IDs for request tracing
   - Premium web interface with dark mode, 3D effects, 5 clinical calculators
   - 104 automated tests with comprehensive coverage

### Clinical Achievements

- **5 validated clinical calculators** implementing peer-reviewed scoring systems (eGFR CKD-EPI 2021, CHA₂DS₂-VASc, MELD-Na, Wells DVT, CURB-65)
- **6 indexed clinical guideline corpora** spanning major domains (cardiology, diabetes, hypertension, anticoagulation, pregnancy, infectious disease)
- **40+ safety rules** covering known drug interactions, allergy cross-reactivities, and condition-specific contraindications
- **30+ knowledge graph nodes** with 20+ typed relationships modeling the clinical domain

### Evaluation Results

The system was evaluated across multiple phases using curated clinical test cases:

| Metric | Phase 0 (LLM-only) | Phase 5 (Full System) | Improvement |
|--------|--------------------|-----------------------|-------------|
| Safety flag detection | None | Dual-layer validation | ∞ |
| Evidence grounding | LLM memory only | RAG + 6 guideline corpora | Grounded |
| Temporal awareness | None | Graph-based decay | Full |
| Human review escalation | None | Multi-dimensional gating | Automatic |
| Drug interaction detection | None | 40+ rules + KG validation | Comprehensive |
| Test coverage | 0 tests | 104 tests | Production-grade |

### Key Insight

The most important finding is that **the symbolic safety layer (Phases 1 + 5) catches errors that the LLM consistently makes**. In evaluation, the LLM occasionally recommends drugs contraindicated by patient allergies or conditions — the rule engine and knowledge graph reliably catch and block these before they reach the clinician. This validates the neuro-symbolic approach: use LLMs for flexible reasoning, but gate every output through formal symbolic validation.

---

# Chapter 5 — Future Work

## Future Areas of Improvement

### 5.1 System Hardening (Phase 6)

The current system lacks several production-hardening features that would be required for real-world deployment:

- **Rate Limiting**: API endpoint rate limiting (e.g., 100 requests/minute per client) to prevent abuse and ensure fair resource allocation
- **Input Sanitization**: Deep validation of patient input beyond Pydantic types — checking for impossible lab values, inconsistent dates, suspiciously large payloads
- **Circuit Breakers**: Automatic fallback behavior when the LLM backend is unavailable or degraded (e.g., return cached results, switch to rule-only mode)
- **Audit Logging**: Immutable audit trail of every analysis, including input data, intermediate results, and final output — required for healthcare compliance (HIPAA, GDPR)
- **Graceful Degradation**: Define behavior when individual pipeline stages fail (currently handles this, but needs formalization)

### 5.2 Graph Neural Networks (Phase 7)

Replace the heuristic-based temporal analysis with trained Graph Neural Networks:

- **Temporal Graph Networks (TGN)**: Learn temporal patterns directly from EHR data rather than relying on hand-coded causal links
- **PyTorch Geometric integration**: Move from NetworkX to GPU-accelerated graph computations
- **Disease progression prediction**: Predict future disease trajectories based on learned patterns
- **Automatic relationship discovery**: Let the GNN discover causal relationships beyond the 16 hand-coded ones
- **Benchmark**: MIMIC-III/IV temporal phenotyping tasks

### 5.3 Production Database Backend (Phase 8)

Migrate from in-memory stores to production databases:

- **Neo4j**: Replace NetworkX knowledge graph with Neo4j for persistent, queryable, ACID-compliant graph storage. Enables Cypher queries and scales to millions of nodes.
- **PostgreSQL**: Replace JSON file feedback storage with relational database. Support multi-user access, complex queries, and migration management (Alembic).
- **Redis**: Replace in-memory embedding cache with distributed Redis cache. Enables cache sharing across multiple API server instances.
- **Qdrant/Weaviate**: Replace numpy vector store with purpose-built vector database. Enables persistent index, approximate nearest neighbor search at scale, and metadata filtering.

### 5.4 Multi-Modal Inputs (Phase 9)

Extend beyond structured EHR data to support:

- **Medical Imaging**: Integrate chest X-ray, CT scan, and MRI interpretation using vision-language models
- **Clinical Notes (NLP)**: Parse unstructured clinical notes using NER (Named Entity Recognition) and relation extraction
- **Waveform Data**: ECG, continuous BP monitoring, and ventilator waveform analysis
- **Genomic Data**: Pharmacogenomic profiles for personalized drug selection

### 5.5 Federated Learning (Phase 10)

Enable privacy-preserving model improvement across institutions:

- **Federated fine-tuning**: Improve LLM reasoning without centralizing patient data
- **Differential privacy**: Mathematical guarantees on patient data protection
- **Cross-institutional validation**: Evaluate model performance across different patient populations
- **Collaborative knowledge graph**: Shared KG updates across hospital networks

### 5.6 Deployment and Scaling

- **Cloud deployment**: Vercel (frontend) + AWS/Azure (API + LLM backend)
- **Horizontal scaling**: Multiple API server instances behind a load balancer
- **Monitoring**: Prometheus + Grafana dashboards for latency, error rates, and safety flag distributions
- **A/B testing**: Compare different prompt versions, model sizes, and safety thresholds in production
- **CI/CD**: Automated testing + deployment pipeline (GitHub Actions)

---

# Chapter 6 — References

## Academic References

1. **Singh, H., Meyer, A. N. D., & Thomas, E. J. (2014)**. The frequency of diagnostic errors in outpatient care: estimations from three large observational studies involving US adult populations. *BMJ Quality & Safety*, 23(9), 727-731.

2. **Bates, D. W., Spell, N., Cullen, D. J., et al. (1997)**. The costs of adverse drug events in hospitalized patients. *JAMA*, 277(4), 307-311.

3. **Lewis, P., Perez, E., Piktus, A., et al. (2020)**. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 9459-9474.

4. **Garcez, A. d'A., Besold, T. R., De Raedt, L., et al. (2019)**. Neural-Symbolic Computing: An Effective Methodology for Principled Integration of Machine Learning and Reasoning. *Journal of Applied Logics*, 6(4), 611-632.

5. **Marcus, G. (2020)**. The Next Decade in AI: Four Steps Towards Robust Artificial Intelligence. *arXiv preprint arXiv:2002.06177*.

6. **Jensen, P. B., Jensen, L. J., & Brunak, S. (2012)**. Mining electronic health records: towards better research applications and clinical care. *Nature Reviews Genetics*, 13(6), 395-405.

7. **Rajkomar, A., Oren, E., Chen, K., et al. (2018)**. Scalable and accurate deep learning with electronic health records. *npj Digital Medicine*, 1(1), 1-10.

8. **Østbye, T., Yarnall, K. S., Krause, K. M., et al. (2005)**. Is there time for management of patients with chronic diseases in primary care? *The Annals of Family Medicine*, 3(3), 209-214.

9. **Singh, H., & Graber, M. L. (2015)**. Improving diagnosis in health care—the next imperative for patient safety. *New England Journal of Medicine*, 373(26), 2493-2495.

10. **Sutton, R. T., Pincock, D., Baumgart, D. C., et al. (2020)**. An overview of clinical decision support systems: benefits, risks, and strategies for success. *npj Digital Medicine*, 3(1), 1-10.

## Clinical Guideline References

11. **Levey, A. S., Stevens, L. A., Schmid, C. H., et al. (2009)**. A New Equation to Estimate Glomerular Filtration Rate. *Annals of Internal Medicine*, 150(9), 604-612. [CKD-EPI equation — basis for eGFR calculator]

12. **Inker, L. A., Eneanya, N. D., Coresh, J., et al. (2021)**. New Creatinine- and Cystatin C-Based Equations to Estimate GFR without Race. *New England Journal of Medicine*, 385(19), 1737-1749. [CKD-EPI 2021 race-free update]

13. **Lip, G. Y. H., Nieuwlaat, R., Pisters, R., et al. (2010)**. Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach. *Chest*, 137(2), 263-272. [CHA₂DS₂-VASc score]

14. **Kim, W. R., Biggins, S. W., Kremers, W. K., et al. (2008)**. Hyponatremia and mortality among patients on the liver-transplant waiting list. *New England Journal of Medicine*, 359(10), 1018-1026. [MELD-Na score]

15. **Wells, P. S., Anderson, D. R., Rodger, M., et al. (2003)**. Evaluation of D-dimer in the diagnosis of suspected deep-vein thrombosis. *New England Journal of Medicine*, 349(13), 1227-1235. [Wells DVT score]

16. **Lim, W. S., van der Eerden, M. M., Laing, R., et al. (2003)**. Defining community acquired pneumonia severity on presentation to hospital: an international derivation and validation study. *Thorax*, 58(5), 377-382. [CURB-65 score]

## Technical References

17. **FastAPI Documentation**. https://fastapi.tiangolo.com/ [Web framework]

18. **Pydantic V2 Documentation**. https://docs.pydantic.dev/latest/ [Data validation]

19. **Sentence-Transformers Documentation**. https://www.sbert.net/ [Embedding models]

20. **NetworkX Documentation**. https://networkx.org/ [Graph library]

21. **Alpine.js Documentation**. https://alpinejs.dev/ [Frontend framework]

22. **TailwindCSS Documentation**. https://tailwindcss.com/ [CSS framework]

23. **Chart.js Documentation**. https://www.chartjs.org/ [Data visualization]

24. **structlog Documentation**. https://www.structlog.org/ [Structured logging]

25. **FHIR R4 Specification**. https://hl7.org/fhir/R4/ [Healthcare data standard]

26. **DrugBank Database**. https://go.drugbank.com/ [Drug interaction data reference]

27. **Agno AI Framework**. https://docs.agno.com/ [Agent framework]

---

<p align="center">
  <strong>NS-TEA v0.5.0</strong> — Neuro-Symbolic Temporal EHR Agent<br/>
  <em>Built with Python 3.12 · FastAPI · Alpine.js · NetworkX · sentence-transformers</em><br/>
  <em>104 tests passing · 6 phases complete · 11 API endpoints</em><br/><br/>
  For research and educational use only. Not intended for clinical deployment without proper validation.
</p>
