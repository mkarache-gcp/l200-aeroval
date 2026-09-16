# AeroEval — Project Context & Session History

*Preserved context for the Google FDE L200 Assessment Agent Project.*  
*Last updated: September 14, 2026*

---

## 1. Project Background & Requirements

- **Program:** Google Cloud Forward Deployed Engineer (FDE) Onboarding — L200 Project.
- **Curriculum:** "AI in 5 Days" (GEAP / ADK Track).
- **Core Objective:** Build, test, and publish an enterprise-grade agent to a public GitHub root repository and submit it for evaluation on the [FDE Project Evaluator](https://fde-project-evaluator-510868799189.us-central1.run.app/login).
- **Evaluation Rubric (Max Score: 95):**
  1. **Tool & Interface Design:** Strongly typed functions, modular schemas, descriptive docstrings.
  2. **Context & Memory:** Multi-turn session state management.
  3. **Orchestration & Logic:** Grounded system instructions, zero hallucinations, structured tool calling.
  4. **Observability & Tracing:** Health probes, structured logs, and telemetry execution traces.
  5. **Infrastructure & CI/CD:** Root project layout, Dockerfile, Terraform IaC, and GitHub Actions CI.

---

## 2. Project Concept: AeroEval

**AeroEval** is an intelligent Hardware Test Engineering Telemetry Evaluation Agent built for aerospace, defense, and robotics applications:
- Test engineers query historical flight test records across drone airframes (`AeroX-1`, `AeroX-2`, `SkyGuardian-Alpha`) and circuit board revisions (`CB-V1.0`, `CB-V2.1-Beta`).
- The agent inspects sensor telemetry logs (`data/telemetry/*.csv`) and mathematically calculates statistical anomalies ($Z\text{-score} > 2.5\sigma$) without fabricating numbers.
- Handles multi-turn inquiries (e.g., asking about a drone's vibration anomalies, then following up on what circuit board was used and whether other flights had issues).

---

## 3. Major Architectural & Technical Decisions

### A. Modular Project Layout
Organized cleanly at the repository root:
```
mkarache-l200/
├── backend/                       # Agent logic, tools, and FastAPI app
│   ├── __init__.py
│   ├── agent.py                   # Vertex AI Agent orchestrator (Gemini 3.8 Flash & Claude Sonnet 4.6)
│   ├── tools.py                   # 3 core ADK Python tools
│   ├── prompt.py                  # Grounded test engineering instructions & guardrails
│   ├── memory.py                  # Multi-turn session context store
│   └── main.py                    # FastAPI server exposing /chat and serving frontend
├── frontend/                      # Web UI for test engineers
│   ├── index.html                 # Clean, centered chat window
│   ├── style.css                  # Dark Google-themed styling
│   └── app.js                     # Turn-by-turn API interaction & model badge
├── infra/                         # Infrastructure as Code (Terraform)
│   ├── main.tf                    # Cloud Run service, Artifact Registry, IAM Service Account
│   ├── variables.tf               # GCP project ID and region variables
│   └── outputs.tf                 # Cloud Run URL output
├── data/                          # Datasets
│   ├── registry.json              # Master flight test metadata index
│   └── telemetry/                 # Telemetry CSV time series
│       ├── flight_101.csv         # Nominal baseline flight (AeroX-1)
│       ├── flight_102.csv         # Motor vibration anomaly (AeroX-2, CB-V2.1-Beta)
│       └── flight_103.csv         # Battery thermal runaway spike (SkyGuardian-Alpha)
├── tests/                         # Test suite
│   ├── test_tools.py              # Unit tests for the 3 telemetry tools
│   ├── test_memory.py             # Session state & context retention tests
│   └── test_api.py                # FastAPI integration tests
├── .github/workflows/ci.yml       # GitHub Actions automated CI
├── Dockerfile                     # Cloud Run container definition
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # Build configuration
└── README.md                      # Architecture documentation & rubric mapping
```

---

### B. Pure IAM & Vertex AI Backend (Zero API Keys)
- **Unified Under Google Cloud Project:** `onboardingproject-507522`.
- **Local Authentication:** Uses standard Google Application Default Credentials (ADC) via:
  ```bash
  gcloud auth application-default login
  ```
- **Cloud Run Authentication:** Uses an attached dedicated IAM Service Account (`aeroeval-agent-sa`) granted `roles/aiplatform.user`.
- **Corporate Workstation mTLS Fix:** On corp machines, disabled mTLS client cert loading via:
  ```ini
  GOOGLE_API_USE_CLIENT_CERTIFICATE=false
  GOOGLE_API_USE_MTLS_ENDPOINT=never
  ```
- **Gemini 3.8 Flash Location:** Configured to `VERTEX_LOCATION=global` on the Gemini Enterprise Agent Platform.
- **Multi-Model Toggling:** Configured support for both **Gemini 3.8 Flash** (`gemini-3.8-flash`) and **Anthropic Claude Sonnet 4.6** (`claude-4-6-sonnet` via Vertex AI Model Garden).

---

### C. The 3 Focused ADK Python Tools (`backend/tools.py`)
1. **`query_flight_metadata(drone_id, board_type, date, registry_path)`**:
   Queries the database / master registry (from Google Cloud Storage `gs://` or local `data/registry.json`) using target hardware constraints (`drone_id`, `board_type`, `date`) and returns matching test IDs and file paths.
2. **`detect_telemetry_anomalies(file_path, metric, threshold)`**:
   Opens a specific telemetry CSV file (streaming directly from GCS or local disk), computes statistical anomalies (Z-score / IQR) for sensor metrics (e.g. `Motor_Vibration_g`, `Battery_Temp_C`, `Voltage_V`), and returns a concise JSON summary of flagged timestamps and values. Keeps bulky raw data out of the LLM context.
3. **`file_incident_ticket(title, severity, drone_id, board_type, test_id, summary, recommendations, confirmed)`**:
   Mocks filing an engineering incident ticket into Jira / Buganizer. Strictly enforces an ADK Human-In-The-Loop (HITL) pre-execution gate: if `confirmed=False`, returns a pending draft state (`REQUIRES_HUMAN_APPROVAL`); if `confirmed=True`, registers the ticket (`TICKET_CREATED`) and returns an assigned ticket ID.

---

### D. Multi-Agent Orchestration & Strategic Model Routing (`backend/agent.py`)
- **`RouterAgent` (Gemini 3.6 Flash via Vertex AI):**
  - Autonomously classifies user intent and conversation context without requiring manual dropdown selection.
  - Classifies queries into `TELEMETRY_EVAL` (AeroEvalAgent) or `DOC_GEN` (DocGenAgent), and detects HITL approvals.
- **`AeroEvalAgent` (Gemini 3.8 Flash via Vertex AI - Preserved Untouched):**
  - Dedicated hardware test evaluation agent executing metadata queries and statistical anomaly detection.
- **`DocGenAgent` (Anthropic Claude 4.6 Sonnet via Vertex AI Model Garden):**
  - Technical documentation and quality engineering agent synthesizing telemetry findings into standardized incident reports.
  - System instruction contains an explicit standard ticket template with full reference example ticket.
- **Human-In-The-Loop (HITL) Validation Hooks:**
  - Enforces pre-execution review on ticket filing.
  - Supports conversational resume ("Approve", "Confirm", "Reject") and programmatic API callback (`POST /approve-action`).

### D. Context & Multi-Turn Memory Architecture (`backend/memory.py`)
- **Unified SessionStore & Backwards Compatibility:** `SessionStore` merges session state modeling and storage management into a single class with `SessionState = SessionStore` alias for full backwards compatibility.
- **Automated History Compaction & Context Truncation:**
  - Configurable threshold (`MAX_HISTORY_TURNS=10`, `RETAIN_RECENT_TURNS=4`).
  - When history exceeds threshold, older turns are distilled into a structured engineering summary turn (`[Session Context Summary: ...]`) + model acknowledgment, preserving prior inquiries and findings.
  - Active hardware context (`active_drone_id`, `active_board_type`, `active_test_id`, `last_file_path`) is permanently tracked and never lost during compaction.
  - Strict alternating turn order (`user` -> `model` -> `user` -> `model`) ensures compatibility with Gemini `client.chats.create(history=...)` and Anthropic Claude.
  - `truncate_context(max_turns=...)` provides sliding-window truncation.
- **Non-Blocking Asynchronous Persistence:**
  - FastAPI `/chat` endpoint uses `BackgroundTasks` (`background_tasks.add_task(session.save)`) so HTTP responses return immediately without waiting for database I/O.
  - `SessionStore` provides `save_async()` (using `asyncio.to_thread`) and `save_in_background()` (using a dedicated `ThreadPoolExecutor`) to eliminate UI blocking.
- **Dual-Layer Persistence:**
  1. In-memory dictionary cache for sub-millisecond local retrieval.
  2. Cloud Firestore integration (`google.cloud.firestore`) persisting full conversational state and active hardware entities (`aeroeval_sessions` collection).
- **Infrastructure as Code:** Declared in `infra/main.tf` (`google_firestore_database` in `FIRESTORE_NATIVE` mode, with `roles/datastore.user` IAM role).

---

### E. Observability & Tracing Architecture (`backend/telemetry.py`)
- **OpenTelemetry SDK Integration:** Uses `opentelemetry.trace` to provide standard distributed tracing across the agent conversation lifecycle.
- **W3C Standards-Compliant Distributed Tracing:**
  - `trace_id`: 32-character hexadecimal string (128-bit) representing the entire conversation turn.
  - `span_id`: 16-character hexadecimal string (64-bit) representing the root turn span.
  - `parent_span_id`: 16-character hexadecimal string linking child spans back to the parent span.
- **Distributed Span Linking in Trace Waterfall:**
  - Each waterfall step (`reasoning_thought`, `tool_call_initiated`, `tool_call_completed`) is recorded as a linked child span containing its own unique `span_id` and explicit `parent_span_id` pointing to the root turn.
- **Google Cloud Trace & Cloud Logging Automatic Correlation:**
  - Emitted structured JSON log includes:
    - `logging.googleapis.com/trace`: `projects/{project_id}/traces/{trace_id}`
    - `logging.googleapis.com/spanId`: `{span_id}`
    - `logging.googleapis.com/trace_sampled`: `true`
  - Allows seamless log-to-trace correlation in the Google Cloud Console.
- **Automated PII Redaction Before Printing:**
  - Implemented `redact_pii` directly inside the telemetry accumulator, executed right before JSON log emission to stdout.
  - Detects and replaces emails (`[REDACTED_EMAIL]`), phone numbers (`[REDACTED_PHONE]`), Social Security Numbers (`[REDACTED_SSN]`), credit card numbers (`[REDACTED_CREDIT_CARD]`), and API/bearer keys (`[REDACTED_API_KEY]`).
  - Recursively sanitizes user queries, model responses, and nested tool waterfall payloads.
- **ADK Callback & Decorator Interception:**
  - `@traced_tool`: Wraps ADK tools with OpenTelemetry spans (`tracer.start_as_current_span`) and captures execution metrics.

---

## 4. Current State & Verification

1. **Live Vertex AI Verification with Gemini 3.8 Flash:**
   Executed live query against `onboardingproject-507522` in `global`:
   - Prompt: *"Were there any motor vibration anomalies on AeroX-2?"*
   - Gemini 3.8 Flash autonomously invoked metadata discovery and anomaly detection.
   - Correctly flagged vibration anomalies on `flight_102` at `14:15:00Z` (2.15g) and `14:16:00Z` (1.94g).

2. **Live Firestore Session & History Persistence (`aeroeval` Database):**
   - Verified active multi-turn sessions logging directly into Firestore collection `aeroeval_sessions`.
   - Each session document retains full turn-by-turn conversational history (`user` and `model` roles) and hardware state tracking across requests.

3. **Live Structured Tracing & Telemetry:**
   - Real-time telemetry emission on turn completion with full trace waterfall (`reasoning_thought`, `tool_call_initiated`, `tool_call_completed`) and metrics.

---

## 5. How to Run Locally

```bash
# 1. Authenticate with GCP ADC (if not already done)
gcloud auth application-default login

# 2. Launch the server (serves both API and Frontend)
python3 -m backend.main
```
Open **[http://localhost:8080](http://localhost:8080)** to chat with the live agent.

---

## 6. Next Steps for Submission
1. Push repository to a public GitHub project root.
2. Deploy the container to Cloud Run using Terraform (`infra/main.tf`) or `gcloud run deploy`.
3. Submit the public Git URL to the [FDE Project Evaluator](https://fde-project-evaluator-510868799189.us-central1.run.app/login).
4. (Optional) Record a 3-minute video walkthrough demonstrating the architecture, code, and live UI demo.

