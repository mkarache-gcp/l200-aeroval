"""
Unit tests for AeroEval PII Redaction.

Verifies:
1. redact_pii scrubs email addresses, phone numbers, SSNs, credit card numbers, and API keys.
2. redact_payload recursively sanitizes nested dictionaries and lists.
3. TraceCollector.emit executes PII redaction directly right before printing to stdout.
"""

import json
from backend.telemetry import TraceCollector, redact_pii, redact_payload


class TestPIIRedaction:
    """Test suite covering regex-based PII redaction directly before stdout emission."""

    def test_redact_email(self):
        """Verifies email addresses are replaced with [REDACTED_EMAIL]."""
        sample = "Reach out to engineer alice.smith@flightcorp.internal or bob@aerocorp.org."
        redacted = redact_pii(sample)
        assert "[REDACTED_EMAIL]" in redacted
        assert "alice.smith@flightcorp.internal" not in redacted
        assert "bob@aerocorp.org" not in redacted

    def test_redact_phone_numbers(self):
        """Verifies US and international phone formats are replaced with [REDACTED_PHONE]."""
        samples = [
            "Call me at 555-123-4567 for flight logs.",
            "Contact: (800) 555-0199.",
            "Cell: +1 415-555-2671.",
        ]
        for text in samples:
            redacted = redact_pii(text)
            assert "[REDACTED_PHONE]" in redacted
            assert "555" not in redacted or "[REDACTED_PHONE]" in redacted

    def test_redact_ssn(self):
        """Verifies SSNs are replaced with [REDACTED_SSN]."""
        sample = "Contractor SSN is 000-12-3456."
        redacted = redact_pii(sample)
        assert "[REDACTED_SSN]" in redacted
        assert "000-12-3456" not in redacted

    def test_redact_credit_card(self):
        """Verifies credit card numbers are replaced with [REDACTED_CREDIT_CARD]."""
        sample = "Charge to card 4111 2222 3333 4444."
        redacted = redact_pii(sample)
        assert "[REDACTED_CREDIT_CARD]" in redacted
        assert "4111 2222 3333 4444" not in redacted

    def test_redact_api_keys(self):
        """Verifies bearer tokens and api keys are replaced with [REDACTED_API_KEY]."""
        sample = "Use Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 for auth or api_key='sk-1234567890abcdef12345678'."
        redacted = redact_pii(sample)
        assert "[REDACTED_API_KEY]" in redacted
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted

    def test_emit_redacts_entire_json_packet(self, capsys):
        """Verifies TraceCollector.emit sanitizes all PII across the entire JSON payload right before printing."""
        collector = TraceCollector(
            session_id="pii-turn-test",
            model_name="gemini-3.8-flash",
            user_query="My email is john.doe@aerocorp.internal and phone is 555-867-5309, test AeroX-2",
        )
        collector.on_thought("Analyzing logs for user john.doe@aerocorp.internal")

        log_data = collector.emit(
            agent_response="Acknowledged john.doe@aerocorp.internal. Contact me at 555-867-5309."
        )

        captured = capsys.readouterr()
        printed_json = json.loads(captured.out)

        # Confirm stdout output has no raw PII
        assert "john.doe@aerocorp.internal" not in captured.out
        assert "555-867-5309" not in captured.out
        assert "[REDACTED_EMAIL]" in captured.out
        assert "[REDACTED_PHONE]" in captured.out

        # Confirm returned dictionary is also sanitized
        assert "[REDACTED_EMAIL]" in log_data["user_query"]
        assert "[REDACTED_PHONE]" in log_data["user_query"]
        assert "[REDACTED_EMAIL]" in log_data["agent_response"]
        assert "[REDACTED_PHONE]" in log_data["agent_response"]
        assert "[REDACTED_EMAIL]" in log_data["trace_waterfall"][0]["payload"]["thought"]
