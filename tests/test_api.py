import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "EMOTIA" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_analyze_frame_no_data():
    response = client.post("/analyze/frame")
    assert response.status_code == 422  # Validation error

# Note: For full testing, would need mock data and trained models
# def test_analyze_frame_with_data():
#     # Mock image data
#     response = client.post("/analyze/frame", files={"image": mock_image})
#     assert response.status_code == 200
#     data = response.json()
#     assert "emotion" in data
#     assert "intent" in data
#     assert "engagement" in data
#     assert "confidence" in data