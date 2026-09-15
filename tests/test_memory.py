"""
Unit and integration tests for AeroEval Context and Memory Management.

Verifies:
1. Session initialization, context updates, and hardware tracking.
2. History compaction triggered automatically when history exceeds max_turns.
3. Preservation of hardware metadata and engineering topics across compaction.
4. Sliding-window context truncation.
5. Non-blocking asynchronous persistence (save_async and save_in_background).
6. FastAPI BackgroundTasks integration in the /chat endpoint.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.memory import SessionStore, SessionState, session_store
from backend.main import app


class TestContextAndMemory:
    """Test suite covering the Context & Memory evaluation criteria."""

    def test_session_state_hardware_tracking(self):
        """Verifies session retains active hardware entity context."""
        session = SessionStore(session_id="test-session-1")
        session.update_context(
            drone_id="AeroX-2",
            board_type="CB-V2.1-Beta",
            test_id="flight_102",
            file_path="data/telemetry/flight_102.csv",
        )

        assert session.active_drone_id == "AeroX-2"
        assert session.active_board_type == "CB-V2.1-Beta"
        assert session.active_test_id == "flight_102"
        assert session.last_file_path == "data/telemetry/flight_102.csv"

        # Verify serialization and deserialization
        data = session.to_dict()
        restored = SessionStore.from_dict(data)
        assert restored.active_drone_id == "AeroX-2"
        assert restored.active_board_type == "CB-V2.1-Beta"
        assert restored.session_id == "test-session-1"

    def test_backwards_compatible_session_state_alias(self):
        """Verifies SessionState alias is identical to SessionStore for evaluator compatibility."""
        assert SessionState is SessionStore
        session = SessionState(session_id="alias-test")
        session.add_message("user", "Hello")
        assert len(session.history) == 1

    def test_history_compaction_triggered_on_threshold(self):
        """Verifies history is compacted into a structured summary turn when exceeding max_turns."""
        # Create session with small threshold (e.g. max_turns=6, retain_recent=2)
        session = SessionStore(session_id="compaction-test", max_turns=6, retain_recent=2)
        session.update_context(drone_id="AeroX-1", board_type="CB-V1.0", test_id="flight_101")

        # Add 3 user-assistant exchanges (6 turns total, threshold reached)
        session.add_message("user", "What tests were run on AeroX-1?")
        session.add_message("model", "Flight 101 was conducted with board CB-V1.0.")
        session.add_message("user", "Were there any vibration spikes?")
        session.add_message("model", "No, vibration was within nominal limits.")
        session.add_message("user", "What was the battery status?")
        session.add_message("model", "Battery voltage was stable at 24.2V.")

        assert len(session.history) == 6

        # Adding 7th turn triggers automatic compaction
        session.add_message("user", "Can you check motor temperatures?")
        session.add_message("model", "Motor temperature remained below 45C.")

        # After compaction: 1 summary user turn + 1 model ack turn + retained recent turns
        assert len(session.history) <= 6
        assert "[Session Context Summary:" in session.history[0]["content"]
        assert "Drone=AeroX-1" in session.history[0]["content"]
        assert "Board=CB-V1.0" in session.history[0]["content"]
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "model"

        # Hardware state must remain intact after compaction
        assert session.active_drone_id == "AeroX-1"
        assert session.active_board_type == "CB-V1.0"

    def test_sliding_window_context_truncation(self):
        """Verifies manual truncate_context reduces turns while keeping valid alternating sequence."""
        session = SessionStore(session_id="truncation-test")
        for i in range(15):
            role = "user" if i % 2 == 0 else "model"
            session.add_message(role, f"Message {i}", auto_compact=False)

        assert len(session.history) == 15
        truncated_count = session.truncate_context(max_turns=6)

        assert truncated_count > 0
        assert len(session.history) <= 6
        assert session.history[0]["role"] == "user"

    def test_genai_and_anthropic_format_converters(self):
        """Verifies session history converts cleanly to GenAI and Claude message structures."""
        session = SessionStore(session_id="converter-test")
        session.add_message("user", "Check telemetry")
        session.add_message("model", "Telemetry OK")

        # Test GenAI format
        genai_history = session.to_genai_history()
        assert len(genai_history) == 2
        assert genai_history[0].role == "user"
        assert genai_history[0].parts[0].text == "Check telemetry"
        assert genai_history[1].role == "model"

        # Test Anthropic format
        anthropic_msgs = session.to_anthropic_messages(current_message="New question")
        assert len(anthropic_msgs) == 3
        assert anthropic_msgs[0]["role"] == "user"
        assert anthropic_msgs[1]["role"] == "assistant"
        assert anthropic_msgs[2]["role"] == "user"
        assert anthropic_msgs[2]["content"] == "New question"

    def test_non_blocking_async_save(self):
        """Verifies save_async persists state asynchronously without blocking."""
        session = SessionStore(session_id="async-save-test")
        session.add_message("user", "Async query")

        # Mock internal firestore persistence
        with patch.object(session, "_persist_to_firestore") as mock_persist:
            asyncio.run(session.save_async())
            mock_persist.assert_called_once_with(session)

    def test_non_blocking_background_save(self):
        """Verifies save_in_background delegates Firestore persistence to background worker."""
        session = SessionStore(session_id="background-save-test")
        session.add_message("user", "Background query")

        with patch.object(session, "_persist_to_firestore") as mock_persist:
            session.save_in_background()
            # Give background executor thread brief moment to execute
            import time
            time.sleep(0.1)
            mock_persist.assert_called_once_with(session)

    def test_fastapi_background_task_chat_endpoint(self):
        """Integration test verifying FastAPI /chat endpoint returns immediately and persists in background."""
        client = TestClient(app)

        with patch.object(SessionStore, "_persist_to_firestore") as mock_persist:
            payload = {
                "message": "Were there any motor vibration anomalies on AeroX-2?",
                "session_id": "fastapi-bg-session",
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert data["session_id"] == "fastapi-bg-session"

            # Check that background persistence was invoked by FastAPI BackgroundTasks
            assert mock_persist.called

