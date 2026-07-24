"""
Unit tests for the FastAPI serving app.

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from serve import app

SAMPLE_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1020.0,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert "model_version" in body


def test_predict_rejects_missing_fields(client):
    incomplete_payload = {"gender": "Female"}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # FastAPI validation error


def test_predict_rejects_invalid_types(client):
    bad_payload = dict(SAMPLE_PAYLOAD)
    bad_payload["tenure"] = "not-a-number"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422