"""
Unit tests for AeroEval ADK Python Tools.

Verifies:
1. query_flight_metadata: Queries registry with drone_id, board_type, and returns matching flights.
2. detect_telemetry_anomalies: Calculates Z-score anomalies on telemetry CSVs and returns summary.
3. Robust error handling on missing files or invalid metrics.
"""

import pytest
from backend.tools import query_flight_metadata, detect_telemetry_anomalies


class TestTelemetryTools:
    """Test suite for ADK Python tools."""

    def test_query_flight_metadata_all(self):
        """Verifies querying flight metadata without filters returns all records."""
        results = query_flight_metadata()
        assert isinstance(results, list)
        assert len(results) >= 3

    def test_query_flight_metadata_by_drone(self):
        """Verifies filtering by drone_id."""
        results = query_flight_metadata(drone_id="AeroX-2")
        assert len(results) >= 2
        for flight in results:
            assert flight["drone_id"] == "AeroX-2"

    def test_query_flight_metadata_by_board(self):
        """Verifies filtering by board_type."""
        results = query_flight_metadata(board_type="CB-V2.1-Beta")
        assert len(results) >= 2
        for flight in results:
            assert flight["circuit_board"] == "CB-V2.1-Beta"

    def test_detect_telemetry_anomalies_nominal(self):
        """Verifies anomaly detection on nominal flight returns 0 anomalies and NOMINAL status."""
        result = detect_telemetry_anomalies(
            file_path="data/telemetry/flight_101.csv",
            metric="Motor_Vibration_g",
            threshold=2.5,
        )
        assert result.get("status") == "NOMINAL"
        assert result.get("anomaly_count") == 0

    def test_detect_telemetry_anomalies_outlier_detected(self):
        """Verifies anomaly detection on flight 102 catches motor vibration spikes."""
        result = detect_telemetry_anomalies(
            file_path="data/telemetry/flight_102.csv",
            metric="Motor_Vibration_g",
            threshold=2.0,
        )
        assert result.get("status") == "ANOMALIES_DETECTED"
        assert result.get("anomaly_count") > 0
        assert len(result.get("anomalies", [])) > 0
        assert result.get("max", 0.0) >= 2.0

    def test_detect_telemetry_anomalies_missing_file(self):
        """Verifies handling of non-existent file path."""
        result = detect_telemetry_anomalies(
            file_path="data/telemetry/non_existent.csv",
            metric="Motor_Vibration_g",
        )
        assert "error" in result
