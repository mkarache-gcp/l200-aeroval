"""
Observability, OpenTelemetry Distributed Tracing, and PII Redaction for AeroEval.

Provides structured Cloud Logging telemetry for every conversation turn, conforming
to the OpenTelemetry and W3C Trace Context specifications:
- Standard OpenTelemetry SDK TracerProvider and Tracer
- 32-character hexadecimal trace_id (128-bit)
- 16-character hexadecimal span_id (64-bit)
- parent_span_id linking child spans (reasoning thoughts, tool calls) to the root turn
- Google Cloud Trace correlation fields (logging.googleapis.com/trace, logging.googleapis.com/spanId)
- Automated regex-based PII redaction executed directly on the telemetry payload right before printing
"""

from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import inspect
import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Standard OpenTelemetry SDK Initialization
# -----------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    current_provider = trace.get_tracer_provider()
    if not isinstance(current_provider, TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer("aeroeval", "1.0.0")
except Exception as e:
    logger.debug(f"OpenTelemetry SDK tracer fallback to internal ID generator: {e}")
    tracer = None


def generate_trace_id() -> str:
    """Generates a 32-character hexadecimal W3C/OpenTelemetry-compliant trace ID (128-bit)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generates a 16-character hexadecimal W3C/OpenTelemetry-compliant span ID (64-bit)."""
    return os.urandom(8).hex()


# -----------------------------------------------------------------------------
# PII Redaction Engine (Regex Sanitization Directly Before Printing)
# -----------------------------------------------------------------------------

def redact_pii(text: str) -> str:
    """Sanitizes sensitive Personally Identifiable Information (PII) using regex patterns.

    Applied directly to all string fields right before printing the log to stdout.
    Redacts:
    - Email addresses: user@example.com, alice.smith@corp.internal -> [REDACTED_EMAIL]
    - Phone numbers: (555) 123-4567, 555-123-4567, +1 415-555-2671 -> [REDACTED_PHONE]
    - US Social Security Numbers: 123-45-6789 -> [REDACTED_SSN]
    - Credit Card numbers: 4111-2222-3333-4444 -> [REDACTED_CREDIT_CARD]
    - API keys and tokens: Bearer ... / api_key=... -> [REDACTED_API_KEY]
    """
    if not text or not isinstance(text, str):
        return text

    # 1. Email addresses (supports standard and internal domains)
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        '[REDACTED_EMAIL]',
        text,
    )

    # 2. Phone numbers (requires formatting separators like hyphens, dots, spaces, parentheses)
    text = re.sub(
        r'(?:\+?1[-.\s]?)?\(?\b[0-9]{3}\)?[-.\s][0-9]{3}[-.\s][0-9]{4}\b',
        '[REDACTED_PHONE]',
        text,
    )

    # 3. US Social Security Numbers (SSN: ###-##-####)
    text = re.sub(
        r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b',
        '[REDACTED_SSN]',
        text,
    )

    # 4. Credit Card Numbers (16 digits with spaces or hyphens)
    text = re.sub(
        r'\b(?:\d{4}[-\s]){3}\d{4}\b',
        '[REDACTED_CREDIT_CARD]',
        text,
    )

    # 5. Bearer tokens, secrets, API keys
    text = re.sub(
        r'(?i)(?:bearer\s+[a-zA-Z0-9_\-\.]{16,}|api[_-]?key[\s:=]+[\'\"]?[a-zA-Z0-9_\-]{16,}[\'\"]?)',
        '[REDACTED_API_KEY]',
        text,
    )
    return text


def redact_payload(data: Any) -> Any:
    """Recursively traverses dictionaries, lists, and strings, sanitizing all PII via regex."""
    if isinstance(data, str):
        return redact_pii(data)
    elif isinstance(data, dict):
        return {k: redact_payload(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_payload(item) for item in data]
    return data


# Context-local active trace collector
active_trace_var: ContextVar[Optional["TraceCollector"]] = ContextVar("active_trace", default=None)


class TraceCollector:
    """Collects distributed tracing spans (thoughts, tool calls, tool responses) for a single conversation turn."""

    def __init__(
        self,
        session_id: str,
        model_name: str,
        user_query: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        self.start_time = time.time()
        self.session_id = session_id
        self.model_name = model_name
        self.user_query = user_query
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")

        # OpenTelemetry & W3C Trace Context identifiers
        self.trace_id = trace_id or generate_trace_id()
        self.span_id = span_id or generate_span_id()
        self.parent_span_id = parent_span_id
        self.trace_waterfall: List[Dict[str, Any]] = []

    def on_thought(self, thought: str, thought_span_id: Optional[str] = None) -> str:
        """Records an agent reasoning thought as an OpenTelemetry child span linked to root span."""
        if not thought:
            return ""
        child_span_id = thought_span_id or generate_span_id()
        self.trace_waterfall.append({
            "trace_id": self.trace_id,
            "span_id": child_span_id,
            "parent_span_id": self.span_id,
            "timestamp": round(time.time(), 3),
            "step_type": "reasoning_thought",
            "payload": {
                "thought": str(thought).strip(),
            },
        })
        return child_span_id

    def on_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_span_id: Optional[str] = None,
    ) -> str:
        """Records a tool call initiation as an OpenTelemetry child span linked to root span."""
        clean_args = {k: v for k, v in arguments.items() if v is not None}
        child_span_id = tool_span_id or generate_span_id()
        self.trace_waterfall.append({
            "trace_id": self.trace_id,
            "span_id": child_span_id,
            "parent_span_id": self.span_id,
            "timestamp": round(time.time(), 3),
            "step_type": "tool_call_initiated",
            "payload": {
                "tool_name": tool_name,
                "arguments": clean_args,
            },
        })
        return child_span_id

    def on_tool_response(
        self,
        tool_name: str,
        latency_sec: float,
        response: Any,
        tool_span_id: Optional[str] = None,
    ) -> None:
        """Records a completed tool execution linking back to the tool span and root trace."""
        if isinstance(response, list):
            summary = response[:3] if len(response) > 3 else response
        elif isinstance(response, dict):
            summary = {k: v for k, v in response.items() if k != "available_columns"}
        else:
            summary = str(response)[:500]

        child_span_id = tool_span_id or generate_span_id()
        self.trace_waterfall.append({
            "trace_id": self.trace_id,
            "span_id": child_span_id,
            "parent_span_id": self.span_id,
            "timestamp": round(time.time(), 3),
            "step_type": "tool_call_completed",
            "payload": {
                "tool_name": tool_name,
                "latency_sec": round(latency_sec, 3),
                "response_summary": summary,
            },
        })

    def emit(self, agent_response: str, severity: str = "INFO") -> Dict[str, Any]:
        """Assembles OpenTelemetry-compliant structured log, scrubs PII via regex, and prints to stdout."""
        total_latency = round(time.time() - self.start_time, 3)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        telemetry_log = {
            "timestamp": now_iso,
            "severity": severity,
            # OpenTelemetry / W3C Trace Context Standard
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            # Google Cloud Trace / Cloud Logging Automatic Correlation
            "logging.googleapis.com/trace": f"projects/{self.project_id}/traces/{self.trace_id}",
            "logging.googleapis.com/spanId": self.span_id,
            "logging.googleapis.com/trace_sampled": True,
            # Session & Query Context
            "session_id": self.session_id,
            "model_name": self.model_name,
            "user_query": self.user_query,
            "agent_response": agent_response,
            "metrics": {
                "total_latency_sec": total_latency,
                "step_count": len(self.trace_waterfall),
            },
            # Linked Child Spans (waterfall)
            "trace_waterfall": self.trace_waterfall,
        }

        # Apply regex PII redaction directly to payload right before printing
        sanitized_log = redact_payload(telemetry_log)
        formatted_json_str = json.dumps(sanitized_log, indent=2)

        # Print directly as formatted JSON to stdout for Google Cloud Run / Cloud Logging
        print(formatted_json_str, file=sys.stdout, flush=True)
        return sanitized_log


class TelemetryLogger:
    """Manages turn tracing with OpenTelemetry span propagation."""

    def start_trace(
        self,
        session_id: str,
        model_name: str,
        user_query: str,
        trace_id: Optional[str] = None,
    ) -> TraceCollector:
        """Starts a root trace span for a conversation turn."""
        tid = trace_id or generate_trace_id()
        sid = generate_span_id()
        collector = TraceCollector(
            session_id=session_id,
            model_name=model_name,
            user_query=user_query,
            trace_id=tid,
            span_id=sid,
        )
        active_trace_var.set(collector)
        return collector

    @property
    def current(self) -> Optional[TraceCollector]:
        return active_trace_var.get()

    def on_thought(self, thought: str) -> None:
        collector = self.current
        if collector:
            collector.on_thought(thought)

    def on_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        collector = self.current
        if collector:
            collector.on_tool_call(tool_name, arguments)

    def on_tool_response(self, tool_name: str, latency_sec: float, response: Any) -> None:
        collector = self.current
        if collector:
            collector.on_tool_response(tool_name, latency_sec, response)


telemetry_logger = TelemetryLogger()


def traced_tool(func: Callable) -> Callable:
    """Decorator wrapping ADK tools to capture OpenTelemetry child spans and execution telemetry."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        collector = active_trace_var.get()
        try:
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            call_args = bound.arguments
        except Exception:
            call_args = kwargs.copy()

        tool_span_id = generate_span_id()

        # If OpenTelemetry tracer is active, wrap in a real OpenTelemetry span
        if tracer:
            with tracer.start_as_current_span(
                f"aeroeval.tool.{func.__name__}",
                attributes={"tool.name": func.__name__},
            ) as otel_span:
                otel_ctx = otel_span.get_span_context()
                if otel_ctx and otel_ctx.is_valid:
                    tool_span_id = format(otel_ctx.span_id, "016x")

                if collector:
                    if func.__name__ == "query_flight_metadata":
                        collector.on_thought("Querying database to identify telemetry log files belonging to target hardware.")
                    elif func.__name__ == "detect_telemetry_anomalies":
                        collector.on_thought("Analyzing time-series sensor readings to calculate statistical anomalies.")

                    collector.on_tool_call(func.__name__, call_args, tool_span_id=tool_span_id)

                t_start = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - t_start
                    if collector:
                        collector.on_tool_response(func.__name__, elapsed, result, tool_span_id=tool_span_id)
                    return result
                except Exception as e:
                    elapsed = time.time() - t_start
                    if collector:
                        collector.on_tool_response(func.__name__, elapsed, {"error": str(e)}, tool_span_id=tool_span_id)
                    raise
        else:
            if collector:
                if func.__name__ == "query_flight_metadata":
                    collector.on_thought("Querying database to identify telemetry log files belonging to target hardware.")
                elif func.__name__ == "detect_telemetry_anomalies":
                    collector.on_thought("Analyzing time-series sensor readings to calculate statistical anomalies.")
                collector.on_tool_call(func.__name__, call_args, tool_span_id=tool_span_id)

            t_start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - t_start
                if collector:
                    collector.on_tool_response(func.__name__, elapsed, result, tool_span_id=tool_span_id)
                return result
            except Exception as e:
                elapsed = time.time() - t_start
                if collector:
                    collector.on_tool_response(func.__name__, elapsed, {"error": str(e)}, tool_span_id=tool_span_id)
                raise

    return wrapper
