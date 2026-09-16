"""
ADK Python Tools for AeroEval Test Engineering Telemetry.

This module provides two focused tools conforming to Google ADK best practices:
1. query_flight_metadata: Queries the database / master registry using user constraints
   (drone_id, board_type, date) and returns matching test IDs and CSV file paths.
   Supports reading from Google Cloud Storage (GCS) or local master registry.
2. detect_telemetry_anomalies: Opens a specific telemetry CSV file (from GCS or local disk),
   runs statistical anomaly detection (Z-score / IQR) on sensor metrics, and returns a
   concise JSON summary of flagged timestamps and values. Keeps raw data out of LLM context.
"""

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional
from backend.telemetry import traced_tool

logger = logging.getLogger(__name__)

# Default path to master registry relative to workspace root
DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "registry.json")


def _read_file_content(path: str) -> Optional[str]:
    """Reads file content from GCS if path is a gs:// URI or bucket is configured, otherwise from local disk."""
    # 1. Direct gs:// URI
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
            client = storage.Client(project=project_id)
            without_prefix = path[5:]
            bucket_name, blob_name = without_prefix.split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text()
            logger.warning(f"GCS blob does not exist: {path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to read from GCS URI {path}: {e}")
            return None

    # 2. If GCS_BUCKET_NAME is set, attempt reading from Cloud Storage
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if bucket_name:
        try:
            from google.cloud import storage
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name)

            # Normalize blob name (e.g. data/telemetry/flight_101.csv -> telemetry/flight_101.csv or registry.json)
            blob_name = path
            if blob_name.startswith("data/"):
                blob_name = blob_name[5:]
            elif "data/" in blob_name:
                blob_name = blob_name.split("data/", 1)[1]

            blob = bucket.blob(blob_name)
            if blob.exists():
                return blob.download_as_text()
        except Exception as e:
            logger.warning(f"Failed to read {path} from GCS bucket {bucket_name}: {e}. Falling back to local disk.")

    # 3. Local disk fallback
    local_path = path
    if not os.path.isabs(local_path):
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        local_path = os.path.join(workspace_root, local_path)

    if os.path.exists(local_path):
        with open(local_path, mode="r", encoding="utf-8") as f:
            return f.read()

    return None


@traced_tool
def query_flight_metadata(
    drone_id: Optional[str] = None,
    board_type: Optional[str] = None,
    date: Optional[str] = None,
    registry_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Queries the database to find flight test records matching the target hardware.

    Returns matching test IDs and their CSV file paths. Supports reading from Google
    Cloud Storage or local registry. Use this tool whenever the user asks about tests
    conducted on a specific drone model, circuit board version, or test date.

    Args:
        drone_id: Optional identifier or model name of the drone (e.g., 'AeroX-1', 'AeroX-2').
        board_type: Optional circuit board version (e.g., 'CB-V1.0', 'CB-V2.1-Beta').
        date: Optional flight test date string in YYYY-MM-DD format (e.g., '2026-09-12').
        registry_path: Optional custom path or gs:// URI to registry JSON file.

    Returns:
        A list of matching flight test metadata dictionaries containing test_id,
        drone_id, circuit_board, test_date, status, and file_path.
    """
    path = registry_path or DEFAULT_REGISTRY_PATH
    content = _read_file_content(path)
    if content is None:
        return [{"error": f"Registry file not found at: {path}"}]

    try:
        records: List[Dict[str, Any]] = json.loads(content)
    except Exception as e:
        return [{"error": f"Failed to parse registry: {str(e)}"}]

    gcs_bucket = os.getenv("GCS_BUCKET_NAME")
    results = []
    for item in records:
        match = True
        if drone_id and drone_id.lower() not in item.get("drone_id", "").lower():
            match = False
        if board_type and board_type.lower() not in item.get("circuit_board", "").lower():
            match = False
        if date and date not in item.get("test_date", ""):
            match = False
        if match:
            raw_path = item.get("file_path", "")
            # If a GCS bucket is active and the path isn't already gs://, format as GCS URI
            if gcs_bucket and not raw_path.startswith("gs://"):
                normalized_blob = raw_path[5:] if raw_path.startswith("data/") else raw_path
                file_path = f"gs://{gcs_bucket}/{normalized_blob}"
            else:
                file_path = raw_path

            results.append({
                "test_id": item.get("test_id"),
                "drone_id": item.get("drone_id"),
                "circuit_board": item.get("circuit_board"),
                "test_date": item.get("test_date"),
                "status": item.get("status"),
                "file_path": file_path,
                "notes": item.get("notes"),
            })

    return results


@traced_tool
def detect_telemetry_anomalies(
    file_path: str,
    metric: str,
    threshold: float = 2.5,
) -> Dict[str, Any]:
    """Opens a specific CSV telemetry file (from GCS or local disk), runs statistical anomaly
    detection over a metric, and returns a summary list of flagged timestamps and values.

    Calculates Z-scores and statistics for the target sensor metric, filtering out
    nominal records to keep heavy raw data out of the LLM and returning only high-value
    statistical summaries and anomalous timestamps.

    Args:
        file_path: Path or gs:// URI to the telemetry CSV file (e.g., 'data/telemetry/flight_102.csv' or 'gs://bucket/telemetry/flight_102.csv').
        metric: Sensor metric column name (e.g., 'Motor_Vibration_g', 'Battery_Temp_C', 'Voltage_V').
        threshold: Standard deviation threshold for Z-score outlier detection (default 2.5).

    Returns:
        A dictionary containing analysis statistics (mean, std dev, min, max, threshold),
        anomaly count, list of flagged timestamps with values, and status.
    """
    content = _read_file_content(file_path)
    if content is None:
        return {"error": f"Telemetry file not found at: {file_path}"}

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return {"error": "Missing dependency 'pandas'. Please install pandas to compute anomalies."}

    try:
        df = pd.read_csv(io.StringIO(content))
        if metric not in df.columns:
            return {
                "error": f"Metric '{metric}' not found in file.",
                "available_columns": list(df.columns),
            }
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            return {"error": f"No numeric values found for metric '{metric}'."}

        mean = float(series.mean())
        std_dev = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        min_val = float(series.min())
        max_val = float(series.max())
        n = len(series)

        if std_dev == 0.0:
            return {
                "file_path": file_path,
                "metric": metric,
                "total_points": n,
                "mean": round(mean, 3),
                "std_dev": 0.0,
                "anomaly_count": 0,
                "anomalies": [],
                "status": "NOMINAL",
            }

        z_scores = (series - mean) / std_dev
        anomalous_indices = df.index[z_scores.abs() > threshold].tolist()

        anomalies = []
        for idx in anomalous_indices:
            row = df.loc[idx]
            val = float(row[metric])
            z = float(z_scores.loc[idx])
            anomalies.append({
                "timestamp": str(row.get("Timestamp", "")),
                "value": round(val, 3),
                "z_score": round(z, 2),
                "deviation_from_mean": round(val - mean, 3),
            })

        return {
            "file_path": file_path,
            "metric": metric,
            "total_points": n,
            "mean": round(mean, 3),
            "std_dev": round(std_dev, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "threshold": threshold,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "status": "ANOMALIES_DETECTED" if anomalies else "NOMINAL",
        }

    except Exception as e:
        return {"error": f"Failed to compute anomalies: {str(e)}"}


from datetime import datetime, timezone
import random


@traced_tool
def file_incident_ticket(
    title: str,
    severity: str,
    drone_id: str,
    board_type: str,
    test_id: str,
    summary: str,
    recommendations: str,
    metric_details: Optional[str] = None,
    confirmed: bool = False,
    draft_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Mocks filing an engineering incident/bug ticket into the issue tracker (e.g. Jira / Buganizer).

    HUMAN-IN-THE-LOOP (HITL) GUARDRAIL:
    Requires explicit human approval before officially creating the ticket.
    - If `confirmed=False`: Generates a pending draft and pauses execution for human review.
    - If `confirmed=True`: Officially creates and registers the ticket in the engineering backlog.

    Args:
        title: Standardized incident title (e.g., '[INCIDENT] AeroX-2 - flight_102: Motor vibration anomalies').
        severity: Issue severity ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
        drone_id: Identified drone airframe model (e.g., 'AeroX-2').
        board_type: Circuit board version (e.g., 'CB-V2.1-Beta').
        test_id: Test flight identifier (e.g., 'flight_102').
        summary: Executive summary of observed telemetry anomalies and test findings.
        recommendations: Concrete engineering corrective actions.
        metric_details: Optional details on Z-score deviations and sensor metrics.
        confirmed: Set to True ONLY after explicit human signoff has been received.
        draft_id: Optional existing draft ID being confirmed.

    Returns:
        A dictionary containing the ticket status ('REQUIRES_HUMAN_APPROVAL' or 'TICKET_CREATED'),
        assigned ticket ID (if confirmed), and complete ticket details.
    """
    assigned_draft_id = draft_id or f"DRAFT-AERO-{random.randint(1000, 9999)}"

    # 1. Pre-execution HITL Gate: If not explicitly confirmed, return pending draft state
    if not confirmed:
        return {
            "status": "REQUIRES_HUMAN_APPROVAL",
            "action": "file_incident_ticket",
            "draft_id": assigned_draft_id,
            "message": (
                f"Incident ticket draft generated. Human-in-the-loop signoff is required before "
                f"submitting to the issue tracker. Please review the details and confirm filing."
            ),
            "draft": {
                "title": title,
                "severity": severity.upper(),
                "drone_id": drone_id,
                "board_type": board_type,
                "test_id": test_id,
                "summary": summary,
                "recommendations": recommendations,
                "metric_details": metric_details,
            },
        }

    # 2. Execution after Human Signoff
    ticket_id = f"AERO-{random.randint(1000, 9999)}"
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "status": "TICKET_CREATED",
        "ticket_id": ticket_id,
        "draft_id": assigned_draft_id,
        "title": title,
        "severity": severity.upper(),
        "drone_id": drone_id,
        "board_type": board_type,
        "test_id": test_id,
        "message": f"Successfully filed incident ticket {ticket_id} with human signoff.",
        "created_at": now_iso,
        "summary": summary,
        "recommendations": recommendations,
        "metric_details": metric_details,
    }


# Backward compatibility aliases
filter_flight_metadata = query_flight_metadata
calculate_telemetry_anomalies = detect_telemetry_anomalies
