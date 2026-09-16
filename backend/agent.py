"""
Vertex AI Multi-Model Agent setup for AeroEval.

Routes ALL model calls exclusively through Google Cloud Vertex AI:
1. Google Gemini 3.8 Flash via google-genai (vertexai=True)
2. Anthropic Claude via Vertex AI Model Garden (AnthropicVertex)

No third-party API keys required; authentication is unified under GCP IAM & ADC.
"""

import asyncio
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

import re
from backend.memory import SessionStore, session_store
from backend.prompt import (
    AEROEVAL_SYSTEM_INSTRUCTION,
    DOC_GEN_SYSTEM_INSTRUCTION,
    ROUTER_SYSTEM_INSTRUCTION,
)
from backend.telemetry import telemetry_logger
from backend.tools import (
    detect_telemetry_anomalies,
    file_incident_ticket,
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


def call_vertex_claude_api(
    project_id: str,
    location: str,
    model: str,
    messages: List[Dict[str, Any]],
    system_instruction: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Invokes Anthropic Claude directly on Google Cloud Vertex AI via rawPredict REST endpoint.

    Uses standard Google Cloud Application Default Credentials (ADC via google-auth) and httpx.
    Eliminates third-party anthropic Python package dependency.
    """
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest
    import httpx

    # Normalize model ID: allow both claude-sonnet-4-6 and claude-4-6-sonnet
    normalized_model = model
    if normalized_model in ("claude-4-6-sonnet", "claude_4_6_sonnet", "claude_sonnet_4_6"):
        normalized_model = "claude-sonnet-4-6"

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = GoogleAuthRequest()
    creds.refresh(auth_req)
    token = creds.token

    loc = location or "global"
    prefix = f"{loc}-" if loc != "global" else ""
    endpoint = f"https://{prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{loc}/publishers/anthropic/models/{normalized_model}:rawPredict"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    body: Dict[str, Any] = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "max_tokens": max_tokens,
        "system": system_instruction,
    }
    if tools:
        body["tools"] = tools

    resp = httpx.post(endpoint, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Vertex AI rawPredict returned HTTP {resp.status_code}: {resp.text}")

    return resp.json()


class AeroEvalAgent:
    """Agent orchestrator querying Gemini and Claude through Google Cloud Vertex AI."""

    def __init__(self):
        self.instruction = AEROEVAL_SYSTEM_INSTRUCTION
        self.tool_map = {t.__name__: t for t in AEROEVAL_TOOLS}
        
        # GCP Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.location = os.getenv("VERTEX_LOCATION", "global")
        self.claude_location = os.getenv("VERTEX_CLAUDE_LOCATION", "global")
        
        # Default models on Vertex AI
        self.default_provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    def chat(
        self,
        message: str,
        session_id: str = "default-session",
        provider: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> str:
        """Processes a chat message via Vertex AI (Gemini or Anthropic Claude)."""
        active_provider = (provider or self.default_provider).lower()
        model_name = self.claude_model if active_provider in ("anthropic", "claude") else self.gemini_model
        agent_name = "DocGenAgent" if active_provider in ("anthropic", "claude") else "AeroEvalAgent"

        if trace is None:
            trace = telemetry_logger.start_trace(
                session_id=session_id,
                model_name=model_name,
                user_query=message,
                agent_name=agent_name,
            )
        else:
            trace.agent_name = agent_name
            trace.model_name = model_name

        session = session_store.get_or_create(session_id)

        if active_provider in ("anthropic", "claude"):
            return self._call_claude_vertex(message, session, trace)
        return self._call_gemini_vertex(message, session, trace)

    async def chat_async(
        self,
        message: str,
        session_id: str = "default-session",
        provider: Optional[str] = None,
    ) -> str:
        """Asynchronously processes a chat message in a worker thread to prevent event loop blocking."""
        return await asyncio.to_thread(self.chat, message, session_id, provider)

    def _call_gemini_vertex(self, message: str, session, trace) -> str:
        """Invokes Gemini 3.8 Flash through Vertex AI with native conversation history."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )

            # Retrieve prior conversation turns formatted for Gemini (compacted if exceeding threshold)
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
            
            # Persist turns in session and asynchronously persist to Firestore
            session.add_message(role="user", content=message)
            session.add_message(role="model", content=reply)
            session.save_in_background()

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
        try:
            # Build full message history including current user turn
            messages = session.to_anthropic_messages(current_message=message)
            response = call_vertex_claude_api(
                project_id=self.project_id,
                location=self.claude_location,
                model=self.claude_model,
                messages=messages,
                system_instruction=self.instruction,
                tools=ANTHROPIC_TOOLS,
            )

            # Tool execution loop
            loop_limit = 5
            while response.get("stop_reason") == "tool_use" and loop_limit > 0:
                loop_limit -= 1
                content_blocks = response.get("content", [])
                tool_uses = [c for c in content_blocks if c.get("type") == "tool_use"]
                tool_results = []
                for tu in tool_uses:
                    fn_name = tu.get("name")
                    fn_args = tu.get("input", {})
                    tool_fn = self.tool_map.get(fn_name)
                    if tool_fn:
                        try:
                            result = tool_fn(**fn_args)
                        except Exception as ex:
                            result = {"error": str(ex)}
                    else:
                        result = {"error": f"Unknown tool: {fn_name}"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.get("id"),
                        "content": json.dumps(result),
                    })

                messages.append({"role": "assistant", "content": content_blocks})
                messages.append({"role": "user", "content": tool_results})

                response = call_vertex_claude_api(
                    project_id=self.project_id,
                    location=self.claude_location,
                    model=self.claude_model,
                    messages=messages,
                    system_instruction=self.instruction,
                    tools=ANTHROPIC_TOOLS,
                )

            content_blocks = response.get("content", [])
            text_blocks = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
            reply = "\n".join(text_blocks) or "Analysis completed."
            session.add_message(role="user", content=message)
            session.add_message(role="assistant", content=reply)
            session.save_in_background()

            if not trace.trace_waterfall:
                trace.on_thought("Formulating direct engineering response to user query.")
            trace.emit(reply)
            return reply
        except Exception as e:
            logger.error(f"Vertex AI Claude call failed: {e}")
            err_msg = (
                f"⚠️ Error calling Claude on Vertex AI (`{self.claude_model}`, project `{self.project_id}`):\n{str(e)}\n\n"
                "Verify Anthropic Claude is enabled in your Vertex AI Model Garden and permissions are granted."
            )
            trace.emit(err_msg, severity="ERROR")
            return err_msg



ANTHROPIC_DOCGEN_TOOLS = [
    {
        "name": "file_incident_ticket",
        "description": "Mocks filing an engineering incident/bug ticket into the issue tracker (e.g. Jira / Buganizer). CRITICAL: Requires human confirmation (confirmed=True) to finalize.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Standardized incident title (e.g. '[INCIDENT] AeroX-2 - flight_102: Motor vibration anomalies')",
                },
                "severity": {
                    "type": "string",
                    "description": "Issue severity: CRITICAL, HIGH, MEDIUM, LOW",
                    "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                },
                "drone_id": {
                    "type": "string",
                    "description": "Drone airframe model name (e.g. AeroX-2)",
                },
                "board_type": {
                    "type": "string",
                    "description": "Circuit board revision (e.g. CB-V2.1-Beta)",
                },
                "test_id": {
                    "type": "string",
                    "description": "Flight test identifier (e.g. flight_102)",
                },
                "summary": {
                    "type": "string",
                    "description": "Executive summary of observed telemetry anomalies and test findings",
                },
                "recommendations": {
                    "type": "string",
                    "description": "Concrete engineering corrective actions",
                },
                "metric_details": {
                    "type": "string",
                    "description": "Optional details on sensor metrics and deviations",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "True ONLY after user provides explicit confirmation/signoff",
                },
                "draft_id": {
                    "type": "string",
                    "description": "Optional existing draft ID being confirmed",
                },
            },
            "required": ["title", "severity", "drone_id", "board_type", "test_id", "summary", "recommendations"],
        },
    },
]


class DocGenAgent:
    """Document Generation Agent powered by Claude 4.6 Sonnet on Vertex AI.

    Synthesizes telemetry findings into standardized incident tickets adhering strictly
    to the engineering ticket structure and enforces Human-In-The-Loop (HITL) signoff
    before officially filing tickets.
    """

    def __init__(self):
        self.instruction = DOC_GEN_SYSTEM_INSTRUCTION
        self.tool_map = {"file_incident_ticket": file_incident_ticket}
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.claude_location = os.getenv("VERTEX_CLAUDE_LOCATION", "global")
        self.model = os.getenv("CLAUDE_DOCGEN_MODEL", "claude-sonnet-4-6")

    def chat(self, message: str, session: SessionStore, trace, is_hitl_approval: bool = False) -> str:
        """Processes document generation inquiries and manages HITL signoff flow."""
        # 1. Check if user is approving or rejecting an active pending ticket draft
        pending = session.get_pending_action()
        is_approval_intent = is_hitl_approval or bool(
            re.search(r'\b(approve|confirm|proceed|yes|file\s+it|submit|sign\s*off)\b', message, re.IGNORECASE)
        )
        is_reject_intent = bool(re.search(r'\b(reject|cancel|discard|abort)\b', message, re.IGNORECASE))

        if pending and is_reject_intent:
            draft_id = pending.get("draft_id", "draft")
            session.clear_pending_action()
            reply = f"❌ Ticket draft `{draft_id}` has been discarded per your instruction. No ticket was filed."
            session.add_message(role="user", content=message)
            session.add_message(role="assistant", content=reply)
            session.save_in_background()
            trace.on_thought(f"Human rejected ticket draft {draft_id}.")
            trace.emit(reply)
            return reply

        if pending and is_approval_intent and pending.get("action_type") == "file_incident_ticket":
            draft_data = pending.get("draft", {})
            draft_id = pending.get("draft_id")
            trace.on_thought(f"Human approved ticket draft {draft_id}. Executing file_incident_ticket tool.")
            result = file_incident_ticket(confirmed=True, draft_id=draft_id, **draft_data)
            session.record_filed_ticket(result)

            reply = (
                f"✅ **Incident Ticket Successfully Filed to Issue Tracker!**\n\n"
                f"- **Ticket ID**: `{result.get('ticket_id')}`\n"
                f"- **Draft ID**: `{result.get('draft_id')}`\n"
                f"- **Title**: {result.get('title')}\n"
                f"- **Severity**: `{result.get('severity')}`\n"
                f"- **Hardware Under Test**: Drone `{result.get('drone_id')}`, Board `{result.get('board_type')}`, Test `{result.get('test_id')}`\n"
                f"- **Status**: Registered in Engineering Issue Tracker with Human Signoff\n\n"
                f"**Summary**:\n{result.get('summary')}\n\n"
                f"**Recommended Actions**:\n{result.get('recommendations')}"
            )
            session.add_message(role="user", content=message)
            session.add_message(role="assistant", content=reply)
            session.save_in_background()
            trace.emit(reply)
            return reply

        # 2. Invoke Claude on Vertex AI for document synthesis via rawPredict
        try:
            # Context handoff from AeroEvalAgent
            context_hint = ""
            if session.active_drone_id or session.active_test_id:
                context_hint = (
                    f"\n[Active Context from Telemetry Analysis: Drone={session.active_drone_id}, "
                    f"Board={session.active_board_type}, Test={session.active_test_id}, "
                    f"TelemetryLog={session.last_file_path}]"
                )

            current_user_content = message + context_hint if context_hint else message
            messages = session.to_anthropic_messages(current_message=current_user_content)

            response = call_vertex_claude_api(
                project_id=self.project_id,
                location=self.claude_location,
                model=self.model,
                messages=messages,
                system_instruction=self.instruction,
                tools=ANTHROPIC_DOCGEN_TOOLS,
            )

            # Tool execution loop
            loop_limit = 5
            while response.get("stop_reason") == "tool_use" and loop_limit > 0:
                loop_limit -= 1
                content_blocks = response.get("content", [])
                tool_uses = [c for c in content_blocks if c.get("type") == "tool_use"]
                tool_results = []
                for tu in tool_uses:
                    fn_name = tu.get("name")
                    fn_args = tu.get("input", {})
                    tool_fn = self.tool_map.get(fn_name)
                    if tool_fn:
                        try:
                            result = tool_fn(**fn_args)
                            if result.get("status") == "REQUIRES_HUMAN_APPROVAL":
                                session.set_pending_action(
                                    action_type="file_incident_ticket",
                                    draft=result.get("draft", {}),
                                    draft_id=result.get("draft_id"),
                                )
                                trace.on_thought(f"HITL Guardrail: Held draft {result.get('draft_id')} for human signoff.")
                        except Exception as ex:
                            result = {"error": str(ex)}
                    else:
                        result = {"error": f"Unknown tool: {fn_name}"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.get("id"),
                        "content": json.dumps(result),
                    })

                messages.append({"role": "assistant", "content": content_blocks})
                messages.append({"role": "user", "content": tool_results})

                response = call_vertex_claude_api(
                    project_id=self.project_id,
                    location=self.claude_location,
                    model=self.model,
                    messages=messages,
                    system_instruction=self.instruction,
                    tools=ANTHROPIC_DOCGEN_TOOLS,
                )

            content_blocks = response.get("content", [])
            text_blocks = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
            reply = "\n".join(text_blocks) or "Incident report drafted."

            # If Claude output formatted report in prose without invoking tool, park draft for HITL approval
            if "[INCIDENT]" in reply and not session.get_pending_action():
                draft_dict = {
                    "title": f"[INCIDENT] {session.active_drone_id or 'AeroX'} - {session.active_test_id or 'test'}: Telemetry anomaly",
                    "severity": "HIGH",
                    "drone_id": session.active_drone_id or "AeroX",
                    "board_type": session.active_board_type or "Unknown",
                    "test_id": session.active_test_id or "Unknown",
                    "summary": reply[:300],
                    "recommendations": "Review flight logs and run bench isolation tests.",
                }
                session.set_pending_action("file_incident_ticket", draft_dict)

            session.add_message(role="user", content=message)
            session.add_message(role="assistant", content=reply)
            session.save_in_background()

            if not trace.trace_waterfall:
                trace.on_thought("DocGenAgent formulated standardized incident report via Claude on Vertex AI.")
            trace.emit(reply)
            return reply

        except Exception as e:
            logger.error(f"DocGenAgent call failed: {e}")
            err_msg = (
                f"⚠️ Error in DocGenAgent calling Claude on Vertex AI (`{self.model}`, project `{self.project_id}`):\n{str(e)}\n\n"
                "Verify Anthropic Claude permissions in Vertex AI Model Garden."
            )
            trace.emit(err_msg, severity="ERROR")
            return err_msg


class RouterAgent:
    """Lightweight Strategic Router powered by Google Gemini 3.6 Flash on Vertex AI.

    Autonomously analyzes engineer intent and conversation context to route
    queries strategically between AeroEvalAgent (Telemetry Analysis) and DocGenAgent
    (Incident Reporting & Ticket Management).
    """

    def __init__(self):
        self.instruction = ROUTER_SYSTEM_INSTRUCTION
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.location = os.getenv("VERTEX_LOCATION", "global")
        self.model = os.getenv("ROUTER_MODEL", "gemini-3.6-flash")

    def route(self, message: str, session: SessionStore, trace) -> Dict[str, Any]:
        """Classifies intent and returns target agent routing decision."""
        # 1. Immediate check for active pending HITL action
        pending = session.get_pending_action()
        if pending and re.search(r'\b(approve|confirm|proceed|yes|file\s+it|submit|reject|cancel|discard)\b', message, re.I):
            reasoning = "User submitted human-in-the-loop signoff/decision on pending ticket draft."
            trace.on_thought(f"RouterAgent classified query: DOC_GEN ({reasoning})")
            return {
                "target_agent": "DOC_GEN",
                "reasoning": reasoning,
                "is_hitl_approval": True,
            }

        # 2. Strategic Routing via Gemini 3.6 Flash on Vertex AI
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
            prompt = (
                f"User Message: {message}\n"
                f"Active Session Context: Drone={session.active_drone_id}, "
                f"Board={session.active_board_type}, Test={session.active_test_id}\n"
                f"Pending Action: {pending.get('draft_id') if pending else 'None'}"
            )

            res = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.instruction,
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )

            parsed = json.loads(res.text)
            target = parsed.get("target_agent", "TELEMETRY_EVAL")
            reasoning = parsed.get("reasoning", "Classified via Gemini 3.6 Flash")
            is_hitl = parsed.get("is_hitl_approval", False)
            trace.on_thought(f"RouterAgent classified query via Gemini 3.6 Flash: {target} ({reasoning})")
            return {
                "target_agent": target,
                "reasoning": reasoning,
                "is_hitl_approval": is_hitl,
            }
        except Exception as e:
            logger.warning(f"RouterAgent Gemini call failed: {e}. Falling back to rule-based routing.")
            is_doc = bool(re.search(
                r'\b(report|ticket|jira|buganizer|document|file\s+a\s+ticket|open\s+a\s+ticket|incident|summary\s+report)\b',
                message,
                re.IGNORECASE,
            ))
            target = "DOC_GEN" if is_doc else "TELEMETRY_EVAL"
            reasoning = f"Rule-based fallback classifier selected {target}"
            trace.on_thought(f"RouterAgent (Fallback): {target} ({reasoning})")
            return {
                "target_agent": target,
                "reasoning": reasoning,
                "is_hitl_approval": False,
            }


class AeroEvalOrchestrator:
    """Multi-Agent Orchestrator managing strategic routing between AeroEvalAgent and DocGenAgent."""

    def __init__(self):
        self.aeroeval_agent = AeroEvalAgent()
        self.docgen_agent = DocGenAgent()
        self.router_agent = RouterAgent()
        self.default_provider = "auto"
        self.project_id = self.aeroeval_agent.project_id

    def chat_orchestrated(
        self,
        message: str,
        session_id: str = "default-session",
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes chat message through strategic router or designated agent."""
        chosen_mode = (provider or self.default_provider).lower()
        session = session_store.get_or_create(session_id)

        # 1. Initialize Turn Trace starting with RouterAgent
        trace = telemetry_logger.start_trace(
            session_id=session_id,
            model_name=self.router_agent.model,
            user_query=message,
            agent_name="RouterAgent",
        )

        # 2. Determine Target Agent
        if chosen_mode in ("docgen", "claude", "anthropic"):
            target_agent = "DOC_GEN"
            routing_info = {
                "router_agent": "RouterAgent",
                "router_model": self.router_agent.model,
                "target_agent": "DOC_GEN",
                "reasoning": "Manual override selected DocGenAgent (Claude 4.6 Sonnet)",
                "is_hitl_approval": False,
            }
        elif chosen_mode in ("telemetry", "gemini"):
            target_agent = "TELEMETRY_EVAL"
            routing_info = {
                "router_agent": "RouterAgent",
                "router_model": self.router_agent.model,
                "target_agent": "TELEMETRY_EVAL",
                "reasoning": "Manual override selected AeroEvalAgent (Gemini 3.8 Flash)",
                "is_hitl_approval": False,
            }
        else:
            # Autonomous Strategic Routing via Gemini 3.6 Flash
            routing_info = self.router_agent.route(message, session, trace)
            target_agent = routing_info.get("target_agent", "TELEMETRY_EVAL")

        # Record Router Agent's result in OpenTelemetry trace waterfall and top-level telemetry
        trace.on_routing(routing_info, router_model=self.router_agent.model)

        # 3. Dispatch to Target Agent with Active Trace
        if target_agent == "DOC_GEN":
            agent_name = "DocGenAgent"
            active_model = self.docgen_agent.model
            trace.agent_name = agent_name
            trace.model_name = active_model
            reply = self.docgen_agent.chat(
                message,
                session,
                trace,
                is_hitl_approval=routing_info.get("is_hitl_approval", False),
            )
        else:
            agent_name = "AeroEvalAgent"
            active_model = self.aeroeval_agent.gemini_model
            trace.agent_name = agent_name
            trace.model_name = active_model
            reply = self.aeroeval_agent.chat(
                message,
                session_id=session_id,
                provider="gemini",
                trace=trace,
            )

        pending = session.get_pending_action()
        return {
            "response": reply,
            "session_id": session_id,
            "provider": chosen_mode,
            "agent": agent_name,
            "model": active_model,
            "routing": routing_info,
            "requires_approval": pending is not None,
            "pending_draft": pending,
        }

    def chat(
        self,
        message: str,
        session_id: str = "default-session",
        provider: Optional[str] = None,
    ) -> str:
        """Backwards-compatible chat interface returning string response directly."""
        res = self.chat_orchestrated(message=message, session_id=session_id, provider=provider)
        return res["response"]

    async def chat_async(
        self,
        message: str,
        session_id: str = "default-session",
        provider: Optional[str] = None,
    ) -> str:
        """Asynchronously processes chat message in worker thread to prevent event loop blocking."""
        return await asyncio.to_thread(self.chat, message, session_id, provider)

    def handle_approval(self, session_id: str, draft_id: str, approved: bool = True) -> Dict[str, Any]:
        """Programmatic Human-In-The-Loop (HITL) approval hook for UI button or API execution."""
        session = session_store.get_or_create(session_id)
        pending = session.get_pending_action()

        if not pending or pending.get("draft_id") != draft_id:
            return {"status": "ERROR", "message": f"No active pending action matching draft_id {draft_id}"}

        if not approved:
            session.clear_pending_action()
            return {"status": "REJECTED", "draft_id": draft_id, "message": "Ticket draft rejected by human engineer."}

        draft_data = pending.get("draft", {})
        result = file_incident_ticket(confirmed=True, draft_id=draft_id, **draft_data)
        session.record_filed_ticket(result)
        return result


def create_agent() -> AeroEvalOrchestrator:
    """Factory function to initialize and return the AeroEval Multi-Agent Orchestrator."""
    return AeroEvalOrchestrator()
