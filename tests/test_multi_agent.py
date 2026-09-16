"""
Unit and integration tests for AeroEval Multi-Agent Orchestration, Strategic Routing,
Document Generation Agent, and Human-In-The-Loop (HITL) Validation Hooks.

Covers:
1. RouterAgent strategic intent classification (TELEMETRY_EVAL vs DOC_GEN).
2. DocGenAgent standardized incident report drafting adherence.
3. ADK Human-in-the-loop (HITL) pre-execution gate in file_incident_ticket (REQUIRES_HUMAN_APPROVAL).
4. HITL execution after human signoff (TICKET_CREATED).
5. Multi-agent state handoff between AeroEvalAgent and DocGenAgent via SessionStore.
6. Programmatic HITL approval endpoint (/approve-action) and /chat multi-agent metadata.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.agent import (
    AeroEvalAgent,
    AeroEvalOrchestrator,
    DocGenAgent,
    RouterAgent,
    create_agent,
)
from backend.main import app
from backend.memory import SessionStore, session_store
from backend.prompt import DOC_GEN_SYSTEM_INSTRUCTION, ROUTER_SYSTEM_INSTRUCTION
from backend.telemetry import telemetry_logger
from backend.tools import file_incident_ticket


class TestMultiAgentArchitecture:
    """Test suite verifying multi-agent patterns, strategic routing, and HITL hooks."""

    def test_router_agent_system_instruction_and_models(self):
        """Verifies RouterAgent is configured for Gemini 3.6 Flash and contains clear routing rules."""
        router = RouterAgent()
        assert "gemini" in router.model.lower()
        assert "TELEMETRY_EVAL" in router.instruction
        assert "DOC_GEN" in router.instruction
        assert "hitl_approval" in router.instruction

    def test_router_fallback_classification(self):
        """Verifies RouterAgent accurately classifies intent via rule-based fallback when offline."""
        router = RouterAgent()
        session = SessionStore(session_id="router-test-1")
        session.update_context(drone_id="AeroX-2", test_id="flight_102")
        trace = telemetry_logger.start_trace(session_id="router-test-1", model_name="gemini-3.6-flash", user_query="test")

        # Telemetry inquiry -> TELEMETRY_EVAL
        res_telemetry = router.route("Were there any motor vibration anomalies on AeroX-2?", session, trace)
        assert res_telemetry["target_agent"] == "TELEMETRY_EVAL"

        # Report / Ticket inquiry -> DOC_GEN
        res_doc = router.route("Generate an incident report and open a Jira ticket for this issue.", session, trace)
        assert res_doc["target_agent"] == "DOC_GEN"

    def test_docgen_agent_system_instruction_has_example_ticket(self):
        """Verifies DocGenAgent prompt contains the mandatory standard ticket structure and reference example."""
        docgen = DocGenAgent()
        assert "claude" in docgen.model.lower()
        assert "[INCIDENT]" in docgen.instruction
        assert "Title" in docgen.instruction
        assert "Severity" in docgen.instruction
        assert "Hardware Under Test" in docgen.instruction
        assert "Anomalous Telemetry Metrics" in docgen.instruction
        assert "Root Cause Hypothesis" in docgen.instruction
        assert "Recommended Engineering Actions" in docgen.instruction
        assert "Human-In-The-Loop" in docgen.instruction or "HITL" in docgen.instruction

    def test_hitl_gate_blocks_ticket_creation_without_confirmation(self):
        """ADK HITL Pre-execution Hook: Verifies file_incident_ticket halts execution when confirmed=False."""
        result = file_incident_ticket(
            title="[INCIDENT] AeroX-2 - flight_102: Motor vibration spike",
            severity="HIGH",
            drone_id="AeroX-2",
            board_type="CB-V2.1-Beta",
            test_id="flight_102",
            summary="Observed 14 vibration anomalies with peak reaching 4.82g.",
            recommendations="Ground AeroX-2 airframe pending structural dampener inspection.",
            confirmed=False,
        )

        assert result["status"] == "REQUIRES_HUMAN_APPROVAL"
        assert result["action"] == "file_incident_ticket"
        assert "draft_id" in result
        assert result["draft"]["drone_id"] == "AeroX-2"
        assert result["draft"]["severity"] == "HIGH"
        assert "ticket_id" not in result  # Crucial: NO ticket filed yet!

    def test_hitl_creates_ticket_after_confirmation(self):
        """ADK HITL Resume Hook: Verifies file_incident_ticket registers ticket when confirmed=True."""
        result = file_incident_ticket(
            title="[INCIDENT] AeroX-2 - flight_102: Motor vibration spike",
            severity="HIGH",
            drone_id="AeroX-2",
            board_type="CB-V2.1-Beta",
            test_id="flight_102",
            summary="Observed 14 vibration anomalies with peak reaching 4.82g.",
            recommendations="Ground AeroX-2 airframe pending structural dampener inspection.",
            confirmed=True,
            draft_id="DRAFT-AERO-9912",
        )

        assert result["status"] == "TICKET_CREATED"
        assert "ticket_id" in result
        assert result["ticket_id"].startswith("AERO-")
        assert result["draft_id"] == "DRAFT-AERO-9912"
        assert result["drone_id"] == "AeroX-2"

    def test_session_store_hitl_lifecycle(self):
        """Verifies SessionStore correctly registers, retrieves, and resolves pending actions."""
        session = SessionStore(session_id="hitl-session-lifecycle")
        draft_payload = {
            "title": "[INCIDENT] SkyGuardian-Alpha - flight_103: Thermal anomaly",
            "severity": "CRITICAL",
            "drone_id": "SkyGuardian-Alpha",
        }

        draft_id = session.set_pending_action("file_incident_ticket", draft_payload)
        assert session.get_pending_action() is not None
        assert session.get_pending_action()["draft_id"] == draft_id
        assert session.get_pending_action()["status"] == "REQUIRES_HUMAN_APPROVAL"

        # Record approval
        filed_ticket = {
            "ticket_id": "AERO-5501",
            "draft_id": draft_id,
            "status": "TICKET_CREATED",
        }
        session.record_filed_ticket(filed_ticket)
        assert session.get_pending_action() is None  # Cleared after resolution
        assert len(session.filed_tickets) == 1
        assert session.filed_tickets[0]["ticket_id"] == "AERO-5501"

    def test_conversational_approval_execution_in_docgen(self):
        """Verifies DocGenAgent detects pending draft and finalizes ticket upon human 'approve' turn."""
        docgen = DocGenAgent()
        session = SessionStore(session_id="conversational-hitl-test")
        session.update_context(drone_id="AeroX-2", board_type="CB-V2.1-Beta", test_id="flight_102")

        # Park draft awaiting human signoff
        session.set_pending_action(
            "file_incident_ticket",
            {
                "title": "[INCIDENT] AeroX-2 - flight_102: Motor vibration threshold exceeded",
                "severity": "HIGH",
                "drone_id": "AeroX-2",
                "board_type": "CB-V2.1-Beta",
                "test_id": "flight_102",
                "summary": "14 vibration anomalies flagged.",
                "recommendations": "Isolate motor mounts.",
            },
            draft_id="DRAFT-AERO-1234",
        )

        trace = telemetry_logger.start_trace(session_id="conversational-hitl-test", model_name="claude-4-6-sonnet", user_query="Approve")
        reply = docgen.chat("Yes, please approve and file the ticket.", session, trace)

        assert "✅" in reply
        assert "Successfully Filed" in reply
        assert "DRAFT-AERO-1234" in reply
        assert session.get_pending_action() is None
        assert len(session.filed_tickets) == 1
        assert session.filed_tickets[0]["ticket_id"].startswith("AERO-")

    def test_conversational_rejection_in_docgen(self):
        """Verifies DocGenAgent discards pending draft if human rejects."""
        docgen = DocGenAgent()
        session = SessionStore(session_id="conversational-reject-test")
        session.set_pending_action("file_incident_ticket", {"title": "Test Draft"}, draft_id="DRAFT-CANCEL-1")

        trace = telemetry_logger.start_trace(session_id="conversational-reject-test", model_name="claude-4-6-sonnet", user_query="Reject")
        reply = docgen.chat("Reject and discard this draft.", session, trace)

        assert "❌" in reply
        assert "discarded" in reply
        assert session.get_pending_action() is None
        assert len(session.filed_tickets) == 0

    def test_multi_agent_orchestrator_routing_metadata(self):
        """Verifies AeroEvalOrchestrator exposes routing and agent provenance in chat_orchestrated."""
        orchestrator = AeroEvalOrchestrator()

        # Mock AeroEvalAgent response for telemetry query
        with patch.object(orchestrator.aeroeval_agent, "chat", return_value="Telemetry nominal."):
            res = orchestrator.chat_orchestrated(
                message="Show tests for AeroX-1",
                session_id="orchestrator-test",
            )
            assert res["agent"] == "AeroEvalAgent"
            assert "AeroEval" in res["agent"]
            assert res["response"] == "Telemetry nominal."

    def test_fastapi_approve_action_endpoint(self):
        """Verifies POST /approve-action endpoint successfully executes HITL resume."""
        client = TestClient(app)
        session = session_store.get_or_create("api-hitl-session")
        draft_id = session.set_pending_action(
            "file_incident_ticket",
            {
                "title": "[INCIDENT] AeroX-2 Vibration Anomaly",
                "severity": "HIGH",
                "drone_id": "AeroX-2",
                "board_type": "CB-V2.1-Beta",
                "test_id": "flight_102",
                "summary": "High vibration during descent.",
                "recommendations": "Inspect dampers.",
            },
            draft_id="DRAFT-AERO-API-1",
        )

        with patch.object(SessionStore, "_persist_to_firestore"):
            # 1. Test Approval
            payload = {
                "session_id": "api-hitl-session",
                "draft_id": draft_id,
                "approved": True,
            }
            res = client.post("/approve-action", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "TICKET_CREATED"
            assert data["ticket_id"].startswith("AERO-")
            assert data["draft_id"] == draft_id
            assert session.get_pending_action() is None
            assert len(session.filed_tickets) == 1

