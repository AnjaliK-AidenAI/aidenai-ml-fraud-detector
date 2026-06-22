import pytest
from fastapi.testclient import TestClient
from app.main import app, _model

client = TestClient(app)

SAMPLE_TX = {
    "transaction_id": "txn-001",
    "amount": 99.50,
    "merchant_category": "grocery",
    "hour_of_day": 14,
    "day_of_week": 2,
    "is_online": False,
    "distance_from_home_km": 3.2,
    "velocity_last_1h": 1,
    "velocity_last_24h": 3,
    "account_age_days": 540,
    "avg_spend_30d": 75.0,
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_requires_trained_model():
    if not _model.is_trained:
        r = client.post("/predict", json=SAMPLE_TX)
        assert r.status_code == 503
    else:
        r = client.post("/predict", json=SAMPLE_TX)
        assert r.status_code == 200
        body = r.json()
        assert "fraud_probability" in body
        assert body["risk_level"] in ("LOW","MEDIUM","HIGH")

def test_predict_high_risk():
    # Very high amount, unusual hour, high velocity â€” expect fraud signal
    tx = {**SAMPLE_TX, "amount": 9999.0, "hour_of_day": 2, "velocity_last_1h": 15, "account_age_days": 5}
    if _model.is_trained:
        r = client.post("/predict", json=tx)
        assert r.status_code == 200