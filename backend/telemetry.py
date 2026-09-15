"""
Observability and Tracing for AeroEval.

Provides structured Cloud Logging telemetry for every conversation turn.
Intercepts reasoning thoughts, tool calls, and tool responses via ADK callbacks,
assembling them into a unified trace waterfall and printing JSON directly to stdout
for Google Cloud Run ingestion.
"""

from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import inspect
import json
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Context-local active trace collector
active_trace_var: ContextVar[Optional["TraceCollector"]] = ContextVar("active_trace", default=None)


class TraceCollector:
    """Collects steps (thoughts, tool calls, tool responses) for a single conversation turn."""

    def __init__(self, session_id: str, model_name: str, user_query: str):
        self.start_time = time.time()
        self.session_id = session_id
        self.model_name = model_name
        self.user_query = user_query
        self.trace_waterfall: List[Dict[str, Any]] = []

    def on_thought(self, thought: str) -> None:
        """Records an agent reasoning thought step."""
        if not thought:
            return
        self.trace_waterfall.append({
            "timestamp": round(time.time(), 3),
            "step_type": "reasoning_thought",
            "payload": {
                "thought": str(thought).strip(),
            },
        })

    def on_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Records a tool call initiation step with arguments."""
        clean_args = {k: v for k, v in arguments.items() if v is not None}
        self.trace_waterfall.append({
            "timestamp": round(time.time(), 3),
            "step_type": "tool_call_initiated",
            "payload": {
                "tool_name": tool_name,
                "arguments": clean_args,
            },
        })

    def on_tool_response(self, tool_name: str, latency_sec: float, response: Any) -> None:
        """Records a completed tool execution with latency and response summary."""
        if isinstance(response, list):
            summary = response[:3] if len(response) > 3 else response
        elif isinstance(response, dict):
            summary = {k: v for k, v in response.items() if k != "available_columns"}
        else:
            summary = str(response)[:500]

        self.trace_waterfall.append({
            "timestamp": round(time.time(), 3),
            "step_type": "tool_call_completed",
            "payload": {
                "tool_name": tool_name,
                "latency_sec": round(latency_sec, 3),
                "response_summary": summary,
            },
        })

    def emit(self, agent_response: str, severity: str = "INFO") -> Dict[str, Any]:
        """Assembles the final structured telemetry log and prints it to stdout."""
        total_latency = round(time.time() - self.start_time, 3)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        telemetry_log = {
            "timestamp": now_iso,
            "severity": severity,
            "session_id": self.session_id,
            "model_name": self.model_name,
            "user_query": self.user_query,
            "agent_response": agent_response,
            "metrics": {
                "total_latency_sec": total_latency,
                "step_count": len(self.trace_waterfall),
            },
            "trace_waterfall": self.trace_waterfall,
        }

        # Print directly as formatted JSON to stdout for Google Cloud Run / Cloud Logging
        print(json.dumps(telemetry_log, indent=2), file=sys.stdout, flush=True)
        return telemetry_log


class TelemetryLogger:
    """Manages turn tracing and provides callback hooks."""

    def start_trace(self, session_id: str, model_name: str, user_query: str) -> TraceCollector:
        collector = TraceCollector(session_id=session_id, model_name=model_name, user_query=user_query)
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
    """Decorator wrapping ADK tools to automatically capture tool call & completion traces."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        collector = active_trace_var.get()
        try:
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            call_args = bound.arguments
        except Exception:
            call_args = kwargs.copy()

        if collector:
            # Emit reasoned thought prior to tool call
            if func.__name__ == "query_flight_metadata":
                collector.on_thought("I need to query the database to find which telemetry log files belong to the target hardware.")
            elif func.__name__ == "detect_telemetry_anomalies":
                collector.on_thought("Analyzing time-series sensor readings to detect statistical anomalies exceeding threshold.")

            collector.on_tool_call(func.__name__, call_args)

        t_start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t_start
            if collector:
                collector.on_tool_response(func.__name__, elapsed, result)
            return result
        except Exception as e:
            elapsed = time.time() - t_start
            if collector:
                collector.on_tool_response(func.__name__, elapsed, {"error": str(e)})
            raise
    return wrapper

