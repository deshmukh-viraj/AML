import pytest
from fastapi.testclient import TestClient
from FastAPI.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def get_payload():
    """asks the api what features it needs"""
    response = client.get("/features")
    assert response.status_code == 200
    features = response.json()["features"]
    return {feature: 0.0 for feature in features}


def test_predict_safe_transaction():
    payload = {"features": get_payload()}
    payload["features"]["anomaly_score"] = 0.01
    payload["features"]["is_toxic_corridor"] = 0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_alert"] is False
    assert data["probability"] < 0.5


def test_predict_fraudulent_transaction():
    payload = {"features": get_payload()}
    payload["features"]["anomaly_score"] = 0.99
    payload["features"]["burst_score_1h"] = 15.0
    payload["features"]["is_toxic_corridor"] = 1

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_alert"] is True
    assert data["alert_tier"] == "TIER_1_EMERGENCY"


def test_predict_bad_ip():
    bad_payload = {"garbage": "data"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_investigate_for_safe_tx():
    payload = {"features": get_payload()}
    payload['features']["anomaly_score"] = 0.01
    
    response = client.post("/investigate", json=payload)
    assert response.status_code == 400
    assert "below the investigation" in response.json()["detail"].lower()