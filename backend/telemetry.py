"""
Observability, OpenTelemetry Distributed Tracing, and Cloud SDP PII Redaction for AeroEval.

Provides structured Cloud Logging telemetry for every conversation turn, conforming
to the OpenTelemetry and W3C Trace Context specifications:
- 32-character hexadecimal trace_id (128-bit)
- 16-character hexadecimal span_id (64-bit)
- parent_span_id linking child spans (reasoning thoughts, tool calls) to the root turn
- Google Cloud Trace correlation fields (logging.googleapis.com/trace, logging.googleapis.com/spanId)
- Enterprise-grade PII scanning & scrubbing via Google Cloud Sensitive Data Protection (SDP / DLP) API
"""

from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import inspect
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# Initialize OpenTelemetry Tracer
try:
    from opentelemetry import trace as otel_trace
    tracer = otel_trace.get_tracer("aeroeval", "1.0.0")
except Exception as e:
    logger.debug(f"OpenTelemetry SDK tracer fallback to internal ID generator: {e}")
    tracer = None

# Initialize Google Cloud Sensitive Data Protection (DLP API)
try:
    from google.cloud import dlp_v2
except ImportError:
    dlp_v2 = None


class SDPRedactor:
    """Google Cloud Sensitive Data Protection (SDP / DLP) API PII Redactor.

    Calls Google Cloud's machine-learning-powered Sensitive Data Protection API
    (DlpServiceClient.deidentify_content) to detect and redact sensitive InfoTypes
    (names, emails, phone numbers, SSNs, credit cards, passport numbers, auth tokens).
    """

    DEFAULT_INFOTYPES = [
        {"name": "EMAIL_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "PERSON_NAME"},
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "CREDIT_CARD_NUMBER"},
        {"name": "PASSPORT"},
        {"name": "AUTH_TOKEN"},
    ]

    def __init__(self, project_id: Optional[str] = None, client: Optional[Any] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.client = client
        if self.client is None:
            self._init_client()

    def _init_client(self) -> None:
        """Initializes the DlpServiceClient."""
        if dlp_v2 is None:
            logger.warning("google-cloud-dlp library is not installed. PII redaction disabled.")
            return

        try:
            self.client = dlp_v2.DlpServiceClient()
            logger.info(f"Connected to Google Cloud Sensitive Data Protection (SDP) for project '{self.project_id}'.")
        except Exception as e:
            logger.warning(f"Failed to connect to Cloud SDP API: {e}. Telemetry will proceed without redaction.")
            self.client = None

    def redact_text(self, text: str) -> str:
        """Calls Google Cloud Sensitive Data Protection API to redact sensitive PII in text."""
        if not text or not isinstance(text, str) or not self.client:
            return text

        if len(text.strip()) == 0:
            return text

        parent = f"projects/{self.project_id}"
        inspect_config = {
            "info_types": self.DEFAULT_INFOTYPES,
            "min_likelihood": dlp_v2.Likelihood.LIKELIHOOD_UNSPECIFIED,
        }
        deidentify_config = {
            "info_type_transformations": {
                "transformations": [
                    {
                        "primitive_transformation": {
                            "replace_with_info_type_config": {}
                        }
                    }
                ]
            }
        }
        item = {"value": text}

        try:
            response = self.client.deidentify_content(
                request={
                    "parent": parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": item,
                }
            )
            return response.item.value
        except Exception as e:
            logger.warning(f"Cloud SDP deidentify_content API call failed: {e}")
            return text

    def redact_payload(self, data: Any) -> Any:
        """Recursively sanitizes nested dictionaries, lists, and strings using Cloud SDP API."""
        if isinstance(data, str):
            return self.redact_text(data)
        elif isinstance(data, dict):
            return {k: self.redact_payload(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.redact_payload(item) for item in data]
        return data


# Global singleton instance of Cloud SDP PII Redactor
sdp_redactor = SDPRedactor()


def generate_trace_id() -> str:
    """Generates a 32-character hexadecimal W3C/OpenTelemetry-compliant trace ID (128-bit)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generates a 16-character hexadecimal W3C/OpenTelemetry-compliant span ID (64-bit)."""
    return os.urandom(8).hex()


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
        clean_thought = sdp_redactor.redact_text(str(thought).strip())
        child_span_id = thought_span_id or generate_span_id()
        self.trace_waterfall.append({
            "trace_id": self.trace_id,
            "span_id": child_span_id,
            "parent_span_id": self.span_id,
            "timestamp": round(time.time(), 3),
            "step_type": "reasoning_thought",
            "payload": {
                "thought": clean_thought,
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
        scrubbed_args = sdp_redactor.redact_payload(clean_args)
        child_span_id = tool_span_id or generate_span_id()
        self.trace_waterfall.append({
            "trace_id": self.trace_id,
            "span_id": child_span_id,
            "parent_span_id": self.span_id,
            "timestamp": round(time.time(), 3),
            "step_type": "tool_call_initiated",
            "payload": {
                "tool_name": tool_name,
                "arguments": scrubbed_args,
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

        scrubbed_summary = sdp_redactor.redact_payload(summary)
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
                "response_summary": scrubbed_summary,
            },
        })

    def emit(self, agent_response: str, severity: str = "INFO") -> Dict[str, Any]:
        """Assembles OpenTelemetry-compliant structured log, sanitizes PII with Cloud SDP, and prints to stdout."""
        total_latency = round(time.time() - self.start_time, 3)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Sanitize user query, model response, and nested trace waterfall using Google Cloud SDP API
        sanitized_query = sdp_redactor.redact_text(self.user_query)
        sanitized_response = sdp_redactor.redact_text(agent_response)
        sanitized_waterfall = sdp_redactor.redact_payload(self.trace_waterfall)

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
            # Session & Query Context (Scrubbed with Cloud SDP)
            "session_id": self.session_id,
            "model_name": self.model_name,
            "user_query": sanitized_query,
            "agent_response": sanitized_response,
            "metrics": {
                "total_latency_sec": total_latency,
                "step_count": len(sanitized_waterfall),
            },
            # Linked Child Spans (waterfall)
            "trace_waterfall": sanitized_waterfall,
        }

        # Print directly as formatted JSON to stdout for Google Cloud Run / Cloud Logging
        print(json.dumps(telemetry_log, indent=2), file=sys.stdout, flush=True)
        return telemetry_log


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

        # If OpenTelemetry tracer is active, wrap in a real OTEL span
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
