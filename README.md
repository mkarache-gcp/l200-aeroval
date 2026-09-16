# 🛰️ AeroEval — Test Engineering Telemetry Agent

**Google FDE L200 Assessment Agent Project**  
*An enterprise-grade agent for querying, inspecting, and evaluating hardware flight test telemetry across aerospace, avionics, and robotics systems.*

---

## 📌 Executive Summary

Modern aerospace and robotics development relies on hundreds of test flight regimes. Test engineers spend countless hours manually correlating hardware revisions (e.g., flight controller board variants, ESC revisions) against massive sensor logs to diagnose component failures.

**AeroEval** solves this by combining the **Google Agent Development Kit (ADK)** with deterministic statistical analysis:
- **Registry Querying**: Locates relevant test flights by airframe model, circuit board revision, and test dates.
- **Telemetry Inspection**: Streams and parses CSV flight logs (battery temperatures, motor vibration, current draw, altitude).
- **Statistical Anomaly Detection**: Automatically calculates Z-scores across sensor feeds to detect outliers and resonance frequencies without hallucinating numbers.
- **Multi-Turn Context**: Preserves active drone and test session memory across multi-turn engineering inquiries.

---

## 🏛️ Architecture Overview

```mermaid
flowchart TD
    User([Test Engineer / Evaluator]) -->|HTTP / Web UI| Frontend[Frontend Web Dashboard]
    Frontend -->|REST API| API[FastAPI Entrypoint (Cloud Run)]

    subgraph AeroEval Backend [AeroEval Agent (Google ADK)]
        API --> AgentCore[ADK Agent Orchestrator]
        AgentCore <--> ContextMem[Session Context Memory]
        AgentCore --> PromptRules[AeroEval Guardrails & Prompt]
        
        AgentCore --> Tool1[filter_flight_metadata]
        AgentCore --> Tool2[read_telemetry_file]
        AgentCore --> Tool3[calculate_telemetry_anomalies]
    end

    subgraph Data Assets
        Tool1 <--> Registry[(data/registry.json)]
        Tool2 <--> TelemetryFiles[(data/telemetry/*.csv)]
        Tool3 <--> TelemetryFiles
    end

    subgraph Infrastructure
        IaC[infra/main.tf] --> CloudRun[Google Cloud Run]
        IaC --> ArtifactReg[Artifact Registry]
        CI[GitHub Actions] --> AutoTests[Pytest Suite]
    end
```

---

## 🎯 Alignment with L200 Evaluation Pillars

| Evaluation Pillar | AeroEval Implementation |
| :--- | :--- |
| **Tool & Interface Design** | 3 focused ADK Python tools (`query_flight_metadata`, `detect_telemetry_anomalies`, `file_incident_ticket`) with strict type hints, Google docstrings, and robust error handling. Master metadata decoupled from telemetry logs. |
| **Context & Memory** | Unified `SessionStore` maintaining multi-turn state (`active_drone_id`, `active_test_id`, `active_board_type`, `pending_action`, `filed_tickets`). Implements automated **history compaction** and **context truncation** to manage LLM context bloat, and **non-blocking asynchronous Firestore persistence** via FastAPI `BackgroundTasks` (`save_async` / `save_in_background`) to eliminate UI blocking. |
| **Orchestration & Logic** | **Collaborative Multi-Agent Architecture** with **Strategic Model Routing** and **Human-in-the-Loop (HITL) Validation Hooks**:<br/>• **`RouterAgent`** (*Google Gemini 3.6 Flash*): Autonomously classifies engineer intent to route queries without requiring manual UI dropdown selection.<br/>• **`AeroEvalAgent`** (*Google Gemini 3.8 Flash*): Dedicated hardware telemetry & anomaly evaluation agent.<br/>• **`DocGenAgent`** (*Anthropic Claude 4.6 Sonnet*): Quality engineering documentation agent adhering strictly to an explicit standard ticket structure with reference example ticket.<br/>• **ADK Human-In-The-Loop (HITL) Hook**: Enforces pre-execution tool interruption on `file_incident_ticket` (`REQUIRES_HUMAN_APPROVAL`), resuming execution only upon human signoff (conversational resume, `/approve-action` API callback, and UI buttons). |
| **Observability & Tracing** | Standardized health probes (`/healthz`), standard **OpenTelemetry SDK** distributed tracing with W3C-compliant 32-hex `trace_id`, 16-hex `span_id`, parent-child span linking (`parent_span_id`), Google Cloud Trace correlation fields, and automated **regex-based PII redaction** (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_SSN]`, `[REDACTED_CREDIT_CARD]`, `[REDACTED_API_KEY]`) directly sanitizing telemetry logs right before printing. |
| **Infrastructure & CI/CD** | Production `Dockerfile` for Cloud Run (GEAP Agent Runtime), Infrastructure as Code via Terraform (`infra/main.tf`), Cloud Storage bucket for datasets, and automated GitHub Actions CI (`.github/workflows/ci.yml`). |

---

## 📂 Project Structure

```
.
├── backend/                       # Agent logic, tools, and API
│   ├── __init__.py
│   ├── tools.py                   # 3 focused ADK tools (metadata query, anomaly detector, ticket filer)
│   ├── prompt.py                  # Grounded test engineering, DocGen & Router instructions
│   ├── agent.py                   # Multi-agent orchestrator (Router Gemini 3.6, AeroEval Gemini 3.8, DocGen Claude 4.6)
│   ├── memory.py                  # Multi-turn session memory, compaction & async Firestore store
│   ├── telemetry.py               # OpenTelemetry distributed tracing & regex PII redaction
│   └── main.py                    # FastAPI server exposing /chat and /approve-action
├── frontend/                      # Web UI for test engineers
│   ├── index.html                 # Single-page interface with strategic routing dropdown
│   ├── style.css                  # Modern Google-themed styling
│   └── app.js                     # Turn-by-turn API interaction & interactive HITL buttons
├── infra/                         # Infrastructure as Code (Terraform)
│   ├── main.tf                    # Cloud Run & Firestore definitions
│   ├── variables.tf               # GCP project and region configuration
│   └── outputs.tf                 # Cloud Run URL outputs
├── data/                          # Telemetry datasets
│   ├── registry.json              # Flight test metadata
│   └── telemetry/                 # Telemetry CSV time series
│       ├── flight_101.csv         # Nominal baseline flight
│       ├── flight_102.csv         # Motor vibration anomaly (CB-V2.1-Beta)
│       └── flight_103.csv         # Battery thermal spike anomaly
├── tests/                         # Automated test suite (34 tests, 100% pass)
│   ├── test_multi_agent.py        # Unit & integration tests for Router, DocGen, and HITL hooks
│   ├── test_memory.py             # Unit & integration tests for context, compaction & async persistence
│   ├── test_redaction.py          # Unit tests for regex PII redaction
│   ├── test_telemetry.py          # Unit tests for OpenTelemetry tracing, span linking & GCP correlation
│   └── test_tools.py              # Unit tests for ADK telemetry tools
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── Dockerfile                     # Cloud Run container definition
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Build configuration
└── README.md                      # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Run Locally with Python

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run automated tests
pytest tests/ -v

# Launch local server
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```
Open [http://localhost:8080](http://localhost:8080) to interact with the web dashboard.

---

### 2. Run with Docker

```bash
# Build Docker image
docker build -t aeroeval:latest .

# Run container locally
docker run -p 8080:8080 aeroeval:latest
```

---

### 3. Deploy to Google Cloud Run (Terraform)

```bash
cd infra
terraform init
terraform apply -var="project_id=YOUR_GCP_PROJECT_ID"
```

---

## 💬 Sample Multi-Turn Conversations

```
User: "Did Drone AeroX-2 have any anomalies during testing?"
AeroEval: "Drone AeroX-2 (Test: flight_102) exhibited 3 Motor_Vibration_g anomalies exceeding the 2.0 Z-score threshold. Peak vibration reached 2.15g at 2026-09-12T14:05:00Z compared to the baseline mean of 0.77g."

User: "What circuit board version was used on that flight?"
AeroEval: "Flight flight_102 on Drone AeroX-2 utilized circuit board revision CB-V2.1-Beta."

User: "Did any other tests with CB-V2.1-Beta experience anomalies?"
AeroEval: "Yes. Drone SkyGuardian-Alpha also tested with CB-V2.1-Beta in flight_103, exhibiting a severe battery thermal runaway anomaly peaking at 67.2°C."
```

---

## 📹 Video Walkthrough
- Public YouTube Walkthrough: *[Link to public video walkthrough will be added here]*

