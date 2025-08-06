"""
Tests for session management API endpoints
"""

import pytest
import uuid
from fastapi.testclient import TestClient


def test_start_session(client: TestClient, clean_session_service):
    """Test starting a new session"""
    response = client.post("/session/start")
    assert response.status_code == 200

    data = response.json()
    assert "session_id" in data

    # Verify session ID is a valid UUID
    session_id = data["session_id"]
    uuid.UUID(session_id)  # Will raise if invalid

    # Verify session exists in service
    assert session_id in clean_session_service.sessions


def test_get_session_summary(client: TestClient, sample_session_id):
    """Test getting session summary"""
    response = client.get(f"/session/{sample_session_id}/summary")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == sample_session_id
    assert "start_time" in data
    assert "reports" in data
    assert "detection_count" in data
    assert "unique_hazards" in data
    assert data["detection_count"] == 0  # New session
    assert data["unique_hazards"] == 0
    assert len(data["reports"]) == 0


def test_get_session_summary_not_found(client: TestClient):
    """Test getting summary for non-existent session"""
    fake_session_id = str(uuid.uuid4())
    response = client.get(f"/session/{fake_session_id}/summary")
    assert response.status_code == 404

    data = response.json()
    assert "Session" in data["detail"]
    assert "not found" in data["detail"]


def test_end_session(client: TestClient, sample_session_id, clean_session_service):
    """Test ending a session"""
    # Verify session exists
    assert sample_session_id in clean_session_service.sessions

    response = client.post(f"/session/{sample_session_id}/end")
    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Session ended"
    assert data["session_id"] == sample_session_id

    # Verify session no longer exists
    assert sample_session_id not in clean_session_service.sessions


def test_end_session_not_found(client: TestClient):
    """Test ending non-existent session"""
    fake_session_id = str(uuid.uuid4())
    response = client.post(f"/session/{fake_session_id}/end")
    assert response.status_code == 404

    data = response.json()
    assert "Session" in data["detail"]
    assert "not found" in data["detail"]


def test_confirm_report_not_found_session(client: TestClient):
    """Test confirming report with non-existent session"""
    fake_session_id = str(uuid.uuid4())
    fake_report_id = str(uuid.uuid4())

    response = client.post(
        f"/session/{fake_session_id}/report/{fake_report_id}/confirm"
    )
    assert response.status_code == 404

    data = response.json()
    assert "Session" in data["detail"]
    assert "not found" in data["detail"]


def test_dismiss_report_not_found_session(client: TestClient):
    """Test dismissing report with non-existent session"""
    fake_session_id = str(uuid.uuid4())
    fake_report_id = str(uuid.uuid4())

    response = client.post(
        f"/session/{fake_session_id}/report/{fake_report_id}/dismiss"
    )
    assert response.status_code == 404

    data = response.json()
    assert "Session" in data["detail"]
    assert "not found" in data["detail"]


def test_multiple_sessions(client: TestClient, clean_session_service):
    """Test creating multiple sessions"""
    # Create first session
    response1 = client.post("/session/start")
    assert response1.status_code == 200
    session_id1 = response1.json()["session_id"]

    # Create second session
    response2 = client.post("/session/start")
    assert response2.status_code == 200
    session_id2 = response2.json()["session_id"]

    # Verify they are different
    assert session_id1 != session_id2

    # Verify both exist
    assert session_id1 in clean_session_service.sessions
    assert session_id2 in clean_session_service.sessions

    # Test getting summaries for both
    summary1 = client.get(f"/session/{session_id1}/summary")
    summary2 = client.get(f"/session/{session_id2}/summary")

    assert summary1.status_code == 200
    assert summary2.status_code == 200
    assert summary1.json()["id"] == session_id1
    assert summary2.json()["id"] == session_id2
