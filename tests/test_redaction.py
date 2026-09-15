"""
Unit and integration tests for Google Cloud Sensitive Data Protection (SDP) PII Redaction.

Verifies:
1. SDPRedactor initializes with standard InfoTypes.
2. Deidentification of sensitive text using Google Cloud SDP (DLP API).
3. Recursive payload redaction for nested dictionaries and lists in telemetry traces.
4. TraceCollector.emit scrubs sensitive PII before printing and serializing telemetry logs.
"""

from unittest.mock import MagicMock, patch
import pytest
from backend.telemetry import SDPRedactor, TraceCollector, sdp_redactor


class TestSDPPIIRedaction:
    """Test suite covering Google Cloud Sensitive Data Protection (SDP / DLP) PII redaction."""

    def test_redactor_initialization_and_infotypes(self):
        """Verifies SDPRedactor has default high-priority InfoTypes configured."""
        redactor = SDPRedactor()
        type_names = [it["name"] for it in redactor.DEFAULT_INFOTYPES]
        assert "EMAIL_ADDRESS" in type_names
        assert "PHONE_NUMBER" in type_names
        assert "PERSON_NAME" in type_names
        assert "US_SOCIAL_SECURITY_NUMBER" in type_names
        assert "CREDIT_CARD_NUMBER" in type_names

    def test_redact_text_with_mocked_dlp_client(self):
        """Verifies redact_text calls DlpServiceClient.deidentify_content and returns scrubbed text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.item.value = "Contact [PERSON_NAME] at [EMAIL_ADDRESS]"
        mock_client.deidentify_content.return_value = mock_response

        redactor = SDPRedactor(client=mock_client)
        result = redactor.redact_text("Contact Alice Smith at alice@company.com")

        assert result == "Contact [PERSON_NAME] at [EMAIL_ADDRESS]"
        mock_client.deidentify_content.assert_called_once()
        call_kwargs = mock_client.deidentify_content.call_args.kwargs
        assert call_kwargs["request"]["item"]["value"] == "Contact Alice Smith at alice@company.com"

    def test_redact_payload_recursive_sanitization(self):
        """Verifies nested dictionaries and lists in tool payloads are recursively sanitized."""
        mock_client = MagicMock()
        def mock_deidentify(request):
            val = request["item"]["value"]
            val = val.replace("alice@company.com", "[EMAIL_ADDRESS]")
            val = val.replace("555-123-4567", "[PHONE_NUMBER]")
            res = MagicMock()
            res.item.value = val
            return res

        mock_client.deidentify_content.side_effect = mock_deidentify
        redactor = SDPRedactor(client=mock_client)

        raw_payload = {
            "operator": "alice@company.com",
            "contact_phone": "555-123-4567",
            "metrics": [1.2, 3.4],
            "metadata": {
                "engineer_email": "alice@company.com",
                "test_id": "flight_102",
            }
        }

        sanitized = redactor.redact_payload(raw_payload)
        assert sanitized["operator"] == "[EMAIL_ADDRESS]"
        assert sanitized["contact_phone"] == "[PHONE_NUMBER]"
        assert sanitized["metadata"]["engineer_email"] == "[EMAIL_ADDRESS]"
        assert sanitized["metadata"]["test_id"] == "flight_102"
        assert sanitized["metrics"] == [1.2, 3.4]

    def test_trace_collector_emit_scrubs_pii(self):
        """Verifies TraceCollector.emit scrubs user query, agent reply, and waterfall using SDP."""
        collector = TraceCollector(
            session_id="pii-session-test",
            model_name="gemini-3.8-flash",
            user_query="My email is john@corp.internal, check vibration",
        )
        collector.on_thought("Analyzing telemetry for john@corp.internal")

        with patch.object(sdp_redactor, "redact_text") as mock_redact_text, \
             patch.object(sdp_redactor, "redact_payload") as mock_redact_payload:
            mock_redact_text.side_effect = lambda t: t.replace("john@corp.internal", "[EMAIL_ADDRESS]")
            mock_redact_payload.side_effect = lambda p: p

            emitted = collector.emit(agent_response="Hello john@corp.internal, vibration is nominal.")
            assert "[EMAIL_ADDRESS]" in emitted["user_query"]
            assert "john@corp.internal" not in emitted["user_query"]
            assert "[EMAIL_ADDRESS]" in emitted["agent_response"]
            assert "john@corp.internal" not in emitted["agent_response"]

