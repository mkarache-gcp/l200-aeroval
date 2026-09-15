"""
Vertex AI Multi-Model Agent setup for AeroEval.

Routes ALL model calls exclusively through Google Cloud Vertex AI:
1. Google Gemini 3.8 Flash via google-genai (vertexai=True)
2. Anthropic Claude via Vertex AI Model Garden (AnthropicVertex)

No third-party API keys required; authentication is unified under GCP IAM & ADC.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
    # If GOOGLE_APPLICATION_CREDENTIALS is set to a placeholder or non-existent file,
    # unset it so google-auth automatically falls back to gcloud user ADC.
    _cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if _cred and not os.path.exists(_cred):
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    
    # Disable mTLS to avoid client certificate loading errors on corp workstations
    os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"
    os.environ["GOOGLE_API_USE_MTLS_ENDPOINT"] = "never"
except ImportError:
    pass

from backend.memory import session_store
from backend.prompt import AEROEVAL_SYSTEM_INSTRUCTION
from backend.telemetry import telemetry_logger
from backend.tools import (
    detect_telemetry_anomalies,
    query_flight_metadata,
)

logger = logging.getLogger(__name__)

# Registered list of ADK python tools
AEROEVAL_TOOLS = [
    query_flight_metadata,
    detect_telemetry_anomalies,
]

# Tool definitions formatted for Claude on Vertex AI
ANTHROPIC_TOOLS = [
    {
        "name": "query_flight_metadata",
        "description": "Queries the database to find flight test records matching target hardware constraints. Returns matching test IDs and local CSV file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drone_id": {
                    "type": "string",
                    "description": "Optional identifier or model name of the drone (e.g. AeroX-1, AeroX-2).",
                },
                "board_type": {
                    "type": "string",
                    "description": "Optional circuit board version (e.g. CB-V1.0, CB-V2.1-Beta).",
                },
                "date": {
                    "type": "string",
                    "description": "Optional test date in YYYY-MM-DD format (e.g. 2026-09-12).",
                },
            },
        },
    },
    {
        "name": "detect_telemetry_anomalies",
        "description": "Opens a specific CSV telemetry file, runs statistical anomaly detection over a metric, and returns a summary list of flagged timestamps and values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the telemetry CSV file.",
                },
                "metric": {
                    "type": "string",
                    "description": "Sensor metric column name (e.g. Motor_Vibration_g, Battery_Temp_C, Voltage_V).",
                },
                "threshold": {
                    "type": "number",
                    "description": "Z-score outlier threshold (default 2.5).",
                },
            },
            "required": ["file_path", "metric"],
        },
    },
]


class AeroEvalAgent:
    """Agent orchestrator querying Gemini and Claude through Google Cloud Vertex AI."""

    def __init__(self):
        self.instruction = AEROEVAL_SYSTEM_INSTRUCTION
        self.tool_map = {t.__name__: t for t in AEROEVAL_TOOLS}
        
        # GCP Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.location = os.getenv("VERTEX_LOCATION", "global")
        self.claude_location = os.getenv("VERTEX_CLAUDE_LOCATION", "us-east5")  # Claude on Vertex commonly hosted in us-east5/us-central1
        
        # Default models on Vertex AI
        self.default_provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-4-6-sonnet")

    def chat(
        self,
        message: str,
        session_id: str = "default-session",
        provider: Optional[str] = None,
    ) -> str:
        """Processes a chat message via Vertex AI (Gemini or Anthropic Claude)."""
        active_provider = (provider or self.default_provider).lower()
        model_name = self.claude_model if active_provider in ("anthropic", "claude") else self.gemini_model
        trace = telemetry_logger.start_trace(session_id=session_id, model_name=model_name, user_query=message)
        session = session_store.get_or_create(session_id)

        if active_provider in ("anthropic", "claude"):
            return self._call_claude_vertex(message, session, trace)
        return self._call_gemini_vertex(message, session, trace)

    def _call_gemini_vertex(self, message: str, session, trace) -> str:
        """Invokes Gemini 3.8 Flash through Vertex AI with native conversation history."""
        if not self._has_gcp_auth():
            return self._local_fallback(message, session, trace, provider="Vertex AI (Gemini)")

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )

            # Retrieve prior conversation turns formatted for Gemini
            genai_history = session.to_genai_history()

            chat_session = client.chats.create(
                model=self.gemini_model,
                config=types.GenerateContentConfig(
                    system_instruction=self.instruction,
                    tools=AEROEVAL_TOOLS,
                    temperature=0.2,
                ),
                history=genai_history if genai_history else None,
            )
            res = chat_session.send_message(message)
            reply = res.text or "Analysis completed."
            
            # Persist turns in session and Firestore
            session.add_message(role="user", content=message)
            session.add_message(role="model", content=reply)
            session_store.save(session)

            if not trace.trace_waterfall:
                trace.on_thought("Formulating direct engineering response to user query.")
            trace.emit(reply)
            return reply
        except Exception as e:
            logger.error(f"Vertex AI Gemini call failed: {e}")
            err_msg = (
                f"⚠️ Error calling Gemini on Vertex AI (`{self.gemini_model}`, project `{self.project_id}`):\n{str(e)}\n\n"
                "Verify that your GCP credentials / service account have the `roles/aiplatform.user` permission."
            )
            trace.emit(err_msg, severity="ERROR")
            return err_msg

    def _call_claude_vertex(self, message: str, session, trace) -> str:
        """Invokes Anthropic Claude through Vertex AI Model Garden with conversation history."""
        if not self._has_gcp_auth():
            return self._local_fallback(message, session, trace, provider="Vertex AI (Claude)")

        try:
            from anthropic import AnthropicVertex

            # Claude on Vertex uses AnthropicVertex with GCP project & region
            client = AnthropicVertex(
                project_id=self.project_id,
                region=self.claude_location,
            )

            # Build full message history including current user turn
            messages = session.to_anthropic_messages(current_message=message)
            response = client.messages.create(
                model=self.claude_model,
                max_tokens=2048,
                system=self.instruction,
                tools=ANTHROPIC_TOOLS,
                messages=messages,
            )

            # Tool execution loop
            while response.stop_reason == "tool_use":
                tool_uses = [c for c in response.content if c.type == "tool_use"]
                tool_results = []
                for tu in tool_uses:
                    tool_fn = self.tool_map.get(tu.name)
                    if tool_fn:
                        try:
                            result = tool_fn(**tu.input)
                        except Exception as ex:
                            result = {"error": str(ex)}
                    else:
                        result = {"error": f"Unknown tool: {tu.name}"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result),
                    })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = client.messages.create(
                    model=self.claude_model,
                    max_tokens=2048,
                    system=self.instruction,
                    tools=ANTHROPIC_TOOLS,
                    messages=messages,
                )

            text_blocks = [c.text for c in response.content if hasattr(c, "text")]
            reply = "\n".join(text_blocks) or "Analysis completed."
            session.add_message(role="user", content=message)
            session.add_message(role="assistant", content=reply)
            session_store.save(session)

            if not trace.trace_waterfall:
                trace.on_thought("Formulating direct engineering response to user query.")
            trace.emit(reply)
            return reply
        except ImportError:
            err_msg = "⚠️ The `anthropic` package is needed for Claude on Vertex. Install with: `pip install anthropic[vertex]`."
            trace.emit(err_msg, severity="ERROR")
            return err_msg
        except Exception as e:
            logger.error(f"Vertex AI Claude call failed: {e}")
            err_msg = (
                f"⚠️ Error calling Claude on Vertex AI (`{self.claude_model}`, project `{self.project_id}`):\n{str(e)}\n\n"
                "Verify Anthropic Claude is enabled in your Vertex AI Model Garden and permissions are granted."
            )
            trace.emit(err_msg, severity="ERROR")
            return err_msg

    def _has_gcp_auth(self) -> bool:
        """Checks whether GCP credentials or IAM environment are configured."""
        # 1. Local Service Account key file
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            return True
        # 2. Local gcloud ADC file (~/.config/gcloud/application_default_credentials.json)
        adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        if os.path.exists(adc_path):
            return True
        # 3. Google Cloud Run / App Engine container runtime (uses attached IAM Service Account)
        if os.getenv("K_SERVICE") or os.getenv("GAE_SERVICE"):
            return True
        return False

    def _local_fallback(self, message: str, session, trace, provider: str) -> str:
        """Deterministic local router used when GCP credentials are not yet configured."""
        msg = message.strip().lower()

        if "aerox-1" in msg or "aerox 1" in msg:
            session.update_context(drone_id="AeroX-1", test_id="flight_101", board_type="CB-V1.0")
            matches = query_flight_metadata(drone_id="AeroX-1")
            flight = matches[0] if matches else {}
            response = (
                f"[{provider} Simulation] Flight **{flight.get('test_id', 'flight_101')}** for **AeroX-1** "
                f"used circuit board **{flight.get('circuit_board', 'CB-V1.0')}**. "
                f"Status: {flight.get('status', 'Completed')}. Telemetry log: `{flight.get('file_path')}`."
            )

        elif "aerox-2" in msg or "aerox 2" in msg:
            session.update_context(drone_id="AeroX-2", test_id="flight_102", board_type="CB-V2.1-Beta")
            anomalies = detect_telemetry_anomalies("data/telemetry/flight_102.csv", "Motor_Vibration_g")
            count = anomalies.get("anomaly_count", 0)
            max_val = anomalies.get("max", 0.0)
            response = (
                f"⚠️ [{provider} Simulation] **AeroX-2 (flight_102)**: Detected **{count} motor vibration anomalies** exceeding threshold! "
                f"Peak vibration reached **{max_val}g** (mean baseline: {anomalies.get('mean', 0.0)}g). "
                f"Board revision was `{session.active_board_type}`."
            )

        elif "skyguardian" in msg:
            session.update_context(drone_id="SkyGuardian-Alpha", test_id="flight_103", board_type="CB-V2.1-Beta")
            anomalies = detect_telemetry_anomalies("data/telemetry/flight_103.csv", "Battery_Temp_C", threshold=1.8)
            count = anomalies.get("anomaly_count", 0)
            max_temp = anomalies.get("max", 0.0)
            response = (
                f"🔥 [{provider} Simulation] **SkyGuardian-Alpha (flight_103)**: Detected **{count} battery thermal anomalies**! "
                f"Peak battery temperature reached **{max_temp}°C**. "
                f"This flight also used board `{session.active_board_type}`."
            )

        elif "circuit board" in msg or "board" in msg:
            if session.active_board_type:
                matches = query_flight_metadata(board_type=session.active_board_type)
                drones = [m.get("drone_id") for m in matches]
                response = (
                    f"[{provider} Simulation] Active board revision **{session.active_board_type}** was tested on flights: "
                    f"{', '.join(drones)}. Both experienced performance anomalies."
                )
            else:
                matches = query_flight_metadata()
                boards = set(m.get("circuit_board") for m in matches if "circuit_board" in m)
                response = f"[{provider} Simulation] Known circuit board revisions in registry: {', '.join(sorted(boards))}."

        else:
            response = (
                f"Hello, I am AeroEval. Currently running in local simulation mode for **{provider}**.\n\n"
                f"To connect live to **Google Cloud Vertex AI** (Project: `{self.project_id}`):\n"
                "1. Add your GCP Service Account JSON key to `.env`:\n"
                "   `GOOGLE_APPLICATION_CREDENTIALS=gcp_credentials.json`\n"
                "   OR authenticate with: `gcloud auth application-default login`\n\n"
                "In simulation mode, you can query:\n"
                "- *'Were there any motor vibration anomalies on AeroX-2?'*\n"
                "- *'Show tests for AeroX-1'*\n"
                "- *'What circuit board was used?'*"
            )

        session.add_message(role="user", content=message)
        session.add_message(role="assistant", content=response)
        session_store.save(session)

        if not trace.trace_waterfall:
            trace.on_thought(f"Handling inquiry via local {provider} engine.")
        trace.emit(response)
        return response


def create_agent() -> AeroEvalAgent:
    """Factory function to initialize and return the AeroEval Agent."""
    return AeroEvalAgent()
