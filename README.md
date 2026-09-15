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
    User([Test Engineer / Evaluator]) -->|Browser UI / HTTP| Frontend[Web UI / FastAPI Entrypoint]
    Frontend -->|POST /chat| API[AeroEval Backend (Cloud Run)]

    subgraph AeroEval Backend [AeroEval Agent (Google ADK)]
        API --> AgentCore[Multi-Model Agent Orchestrator]
        AgentCore <--> ContextMem[Session Context Memory]
        AgentCore --> PromptRules[System Instructions & Guardrails]
        AgentCore --> Telemetry[ADK Telemetry & Tracing Callbacks]
        
        AgentCore --> Tool1[query_flight_metadata]
        AgentCore --> Tool2[detect_telemetry_anomalies]
    end

    subgraph Foundation Models [Vertex AI / GEAP]
        AgentCore <--> Gemini[Gemini 3.8 Flash @ global]
        AgentCore <--> Claude[Claude Sonnet 4.6 @ us-east5]
    end

    subgraph Cloud Persistence [GCP Serverless State & Data]
        ContextMem <--> Firestore[(Firestore Native: aeroeval)]
        Tool1 <--> GCS[(Cloud Storage: aeroeval-data)]
        Tool2 <--> GCS
        Telemetry --> CloudLogging[Google Cloud Logging]
    end

    subgraph Infrastructure
        IaC[infra/main.tf] --> CloudRun[Google Cloud Run]
        IaC --> Firestore
        IaC --> GCS
        CI[GitHub Actions] --> Validation[Build & Syntax Pipeline]
    end
```

---

## 🎯 Alignment with L200 Evaluation Pillars

| Evaluation Pillar | AeroEval Implementation |
| :--- | :--- |
| **Tool & Interface Design** | 2 focused ADK Python tools (`query_flight_metadata`, `detect_telemetry_anomalies`) with strict type hints, Google docstrings, and robust error handling. Master metadata decoupled from telemetry logs. |
| **Context & Memory** | `SessionStore` with native Gemini `client.chats.create(history=...)` multi-turn history injection, backed by persistent Google Cloud Firestore (`aeroeval` database). |
| **Orchestration & Logic** | Multi-model routing (Gemini 3.8 Flash & Claude Sonnet 4.6 on Vertex AI) with grounded system prompt, strict anti-hallucination guardrails, and autonomous tool calling. |
| **Observability & Tracing** | Standardized health probes (`/healthz`) and structured JSON telemetry logs with ADK callback interception (`@agent.on_thought`, `@agent.on_tool_call`, `@agent.on_tool_response`) outputting trace waterfalls directly to stdout for Cloud Logging. |
| **Infrastructure & CI/CD** | Production `Dockerfile` for Cloud Run (GEAP Agent Runtime), Infrastructure as Code via Terraform (`infra/main.tf`), Cloud Storage bucket for datasets, and automated GitHub Actions CI (`.github/workflows/ci.yml`). |

---

## 📂 Project Structure

```
.
├── backend/                       # Agent logic, tools, and API
│   ├── __init__.py
│   ├── tools.py                   # 2 focused ADK tools (metadata query, anomaly detector)
│   ├── prompt.py                  # System prompt and guardrails
│   ├── agent.py                   # Multi-model Vertex AI orchestrator
│   ├── memory.py                  # Multi-turn session memory & Firestore store
│   ├── telemetry.py               # Structured Cloud Logging trace collector
│   └── main.py                    # FastAPI server & static UI mount
├── frontend/                      # Web UI for test engineers
│   ├── index.html                 # Single-page interface
│   ├── style.css                  # Modern Google-themed styling
│   └── app.js                     # Turn-by-turn API interaction
├── infra/                         # Infrastructure as Code (Terraform)
│   ├── main.tf                    # Cloud Run & Artifact Registry definitions
│   ├── variables.tf               # GCP project and region configuration
│   └── outputs.tf                 # Cloud Run URL outputs
├── data/                          # Telemetry datasets
│   ├── registry.json              # Flight test metadata
│   └── telemetry/                 # Telemetry CSV time series
│       ├── flight_101.csv         # Nominal baseline flight
│       ├── flight_102.csv         # Motor vibration anomaly (CB-V2.1-Beta)
│       └── flight_103.csv         # Battery thermal spike anomaly
├── tests/                         # Automated tests
│   └── test_tools.py              # Unit tests for the 3 tools
├── .github/workflows/ci.yml       # GitHub Actions workflow
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

