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
| **Tool & Interface Design** | 2 focused ADK Python tools (`query_flight_metadata`, `detect_telemetry_anomalies`) with strict type hints, Google docstrings, and robust error handling. Master metadata decoupled from telemetry logs. |
| **Context & Memory** | Unified `SessionStore` maintaining multi-turn state (`active_drone_id`, `active_test_id`, `active_board_type`). Implements automated **history compaction** and **context truncation** to manage LLM context bloat, and **non-blocking asynchronous Firestore persistence** via FastAPI `BackgroundTasks` (`save_async` / `save_in_background`) to eliminate UI blocking. |
| **Orchestration & Logic** | Multi-model routing (Gemini 3.8 Flash & Claude Sonnet 4.6 on Vertex AI) with grounded system prompt, strict anti-hallucination guardrails, and autonomous tool calling. |
| **Observability & Tracing** | Standardized health probes (`/healthz`) and structured JSON telemetry logs with ADK callback interception outputting trace waterfalls directly to stdout for Cloud Logging. |
| **Infrastructure & CI/CD** | Production `Dockerfile` for Cloud Run (GEAP Agent Runtime), Infrastructure as Code via Terraform (`infra/main.tf`), Cloud Storage bucket for datasets, and automated GitHub Actions CI (`.github/workflows/ci.yml`). |

---

## 📂 Project Structure

```
.
├── backend/                       # Agent logic, tools, and API
│   ├── __init__.py
│   ├── tools.py                   # 2 focused ADK tools (metadata query, anomaly detector)
│   ├── prompt.py                  # Grounded test engineering instructions & guardrails
│   ├── agent.py                   # Multi-model Vertex AI orchestrator (Gemini 3.8 & Claude 4.6)
│   ├── memory.py                  # Multi-turn session memory, compaction & async Firestore store
│   ├── telemetry.py               # Structured Cloud Logging trace collector
│   └── main.py                    # FastAPI server exposing /chat with BackgroundTasks
├── frontend/                      # Web UI for test engineers
│   ├── index.html                 # Single-page interface
│   ├── style.css                  # Modern Google-themed styling
│   └── app.js                     # Turn-by-turn API interaction
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
├── tests/                         # Automated test suite
│   ├── test_memory.py             # Unit & integration tests for context, compaction & async persistence
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

