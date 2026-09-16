"""
Unit tests for AeroEval OpenTelemetry Distributed Tracing & Span Linking.

Verifies:
1. Standards-compliant 32-character hexadecimal trace_id and 16-character span_id.
2. Distributed tracing span linking: child spans (thoughts, tool calls, tool responses)
   contain parent_span_id matching the root turn span_id.
3. Top-level telemetry emission includes Google Cloud Trace correlation fields
   (logging.googleapis.com/trace and logging.googleapis.com/spanId).
4. Tool decoration with @traced_tool creates linked OpenTelemetry child spans.
"""

import json
import re
import pytest
from backend.telemetry import (
    TelemetryLogger,
    TraceCollector,
    generate_trace_id,
    generate_span_id,
    telemetry_logger,
    traced_tool,
)


class TestOpenTelemetryTracing:
    """Test suite covering OpenTelemetry distributed tracing and span linking."""

    def test_w3c_identifier_format(self):
        """Verifies trace_id is 32-hex (128-bit) and span_id is 16-hex (64-bit)."""
        trace_id = generate_trace_id()
        span_id = generate_span_id()

        assert len(trace_id) == 32
        assert re.fullmatch(r"^[0-9a-fA-F]{32}$", trace_id) is not None

        assert len(span_id) == 16
        assert re.fullmatch(r"^[0-9a-fA-F]{16}$", span_id) is not None

    def test_trace_collector_span_linking(self):
        """Verifies that child steps (thoughts, tool calls) link to root span_id via parent_span_id."""
        collector = TraceCollector(
            session_id="test-session-otel",
            model_name="gemini-3.8-flash",
            user_query="Check motor vibration on AeroX-2",
        )

        root_trace_id = collector.trace_id
        root_span_id = collector.span_id

        # Record thought
        thought_span_id = collector.on_thought("Reasoning about flight telemetry anomaly.")
        assert thought_span_id is not None
        assert len(thought_span_id) == 16

        # Record tool call and response
        tool_span_id = collector.on_tool_call(
            tool_name="detect_telemetry_anomalies",
            arguments={"file_path": "data/telemetry/flight_102.csv", "metric": "Motor_Vibration_g"},
        )
        collector.on_tool_response(
            tool_name="detect_telemetry_anomalies",
            latency_sec=0.015,
            response={"status": "ANOMALIES_DETECTED", "anomaly_count": 2},
            tool_span_id=tool_span_id,
        )

        waterfall = collector.trace_waterfall
        assert len(waterfall) == 3

        # Every child step must inherit root trace_id and point to root span_id as parent_span_id
        for step in waterfall:
            assert step["trace_id"] == root_trace_id
            assert step["parent_span_id"] == root_span_id
            assert len(step["span_id"]) == 16

        # Step 0: Thought
        assert waterfall[0]["step_type"] == "reasoning_thought"
        assert waterfall[0]["span_id"] == thought_span_id

        # Step 1: Tool call initiation
        assert waterfall[1]["step_type"] == "tool_call_initiated"
        assert waterfall[1]["span_id"] == tool_span_id

        # Step 2: Tool call response
        assert waterfall[2]["step_type"] == "tool_call_completed"
        assert waterfall[2]["span_id"] == tool_span_id

    def test_emit_opentelemetry_and_gcp_trace_fields(self, capsys):
        """Verifies emitted log contains OpenTelemetry and GCP Cloud Trace fields."""
        collector = TraceCollector(
            session_id="emit-test-session",
            model_name="gemini-3.8-flash",
            user_query="Show flights",
        )
        collector.on_thought("Querying registry")
        log_data = collector.emit(agent_response="Found 3 flights.", severity="INFO")

        # Capture printed stdout
        captured = capsys.readouterr()
        printed_json = json.loads(captured.out)

        # Validate OpenTelemetry fields
        assert printed_json["trace_id"] == collector.trace_id
        assert printed_json["span_id"] == collector.span_id

        # Validate GCP Cloud Trace correlation fields
        assert "logging.googleapis.com/trace" in printed_json
        assert printed_json["logging.googleapis.com/trace"].endswith(collector.trace_id)
        assert printed_json["logging.googleapis.com/spanId"] == collector.span_id
        assert printed_json["logging.googleapis.com/trace_sampled"] is True

        # Validate child spans in waterfall
        assert len(printed_json["trace_waterfall"]) == 1
        assert printed_json["trace_waterfall"][0]["parent_span_id"] == collector.span_id

    def test_traced_tool_decorator_propagates_spans(self):
        """Verifies @traced_tool wraps execution and links child spans in the active trace."""
        @traced_tool
        def mock_analysis_tool(flight_id: str) -> dict:
            return {"status": "ok", "flight_id": flight_id}

        collector = telemetry_logger.start_trace(
            session_id="mock-session",
            model_name="mock-model",
            user_query="Run mock analysis",
        )

        result = mock_analysis_tool("flight_999")
        assert result["flight_id"] == "flight_999"

        # The decorator should have recorded thought, tool call, and tool response
        waterfall = collector.trace_waterfall
        assert len(waterfall) >= 2  # tool_call_initiated and tool_call_completed

        tool_calls = [s for s in waterfall if s["step_type"] == "tool_call_initiated"]
        tool_responses = [s for s in waterfall if s["step_type"] == "tool_call_completed"]

        assert len(tool_calls) == 1
        assert len(tool_responses) == 1
        assert tool_calls[0]["span_id"] == tool_responses[0]["span_id"]
        assert tool_calls[0]["parent_span_id"] == collector.span_id

    def test_agent_name_and_router_classification_in_telemetry(self, capsys):
        """Verifies agent_name is initiated/logged and router classification appears in waterfall & top-level."""
        collector = TraceCollector(
            session_id="multi-agent-trace-session",
            model_name="gemini-3.6-flash",
            user_query="Generate report for AeroX-2",
            agent_name="RouterAgent",
        )

        assert collector.agent_name == "RouterAgent"

        # Record router agent decision
        routing_info = {
            "target_agent": "DOC_GEN",
            "reasoning": "User requested incident report and ticket drafting",
            "is_hitl_approval": False,
        }
        router_span_id = collector.on_routing(routing_info, router_model="gemini-3.6-flash")

        assert len(router_span_id) == 16
        assert len(collector.trace_waterfall) == 1
        router_step = collector.trace_waterfall[0]
        assert router_step["step_type"] == "router_classification"
        assert router_step["payload"]["target_agent"] == "DOC_GEN"
        assert router_step["payload"]["router_model"] == "gemini-3.6-flash"

        # Switch active agent to DocGenAgent and emit
        collector.agent_name = "DocGenAgent"
        collector.model_name = "claude-4-6-sonnet"
        log_data = collector.emit("Incident report drafted.")

        captured = capsys.readouterr()
        printed = json.loads(captured.out)

        # Validate agent_name in top-level log
        assert printed["agent_name"] == "DocGenAgent"
        assert printed["model_name"] == "claude-4-6-sonnet"

        # Validate router results in top-level log and waterfall
        assert "routing" in printed
        assert printed["routing"]["target_agent"] == "DOC_GEN"
        assert len(printed["trace_waterfall"]) == 1
        assert printed["trace_waterfall"][0]["step_type"] == "router_classification"
        assert printed["trace_waterfall"][0]["payload"]["router_agent"] == "RouterAgent"

