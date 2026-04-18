# ============================================================
# Tests for Credit Card Propensity API
# Run: pytest tests/test_api.py -v
# ============================================================

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

FEATURE_NAMES = [
    "age", "gender", "tenure_to_bank", "avg_casa_this_m",
    "avg_bal_amt_6mtd_fcy_casa", "avg_bal_amt_ytd_fcy_casa",
    "cr_amt_mtd_fcy_casa", "dr_amt_mtd_fcy_casa",
    "cr_amt_qtd_fcy_casa", "dr_amt_qtd_fcy_casa",
    "avg_loan_lmt", "max_loan_lmt", "max_loan_dsbr_amt",
    "avg_loan_duration", "avg_td_last_2m", "sum_td_this_m",
    "debit_credit_ratio", "cash_pressure", "pressure_score",
    "spend_velocity", "txn_velocity", "momentum_score",
    "casa_trend", "balance_change_ratio", "utilization_ratio",
    "loan_gap", "loan_pressure_ratio", "tenure_x_util",
    "age_x_spend", "eligible_by_age", "high_spend_flag",
    "low_balance_flag", "active_txn_flag",
]

SAMPLE_FEATURES = [
    46, 1, 8.32, 3310613.69, 3367367.83, 2348056.57,
    3053211.37, 3847988.53, 7404976.06, 13258901.81,
    78920973.24, 119569150.41, 48365930.36, 70.93,
    2950061.74, 1766109.95, 1.26, 0.61, 5853925.75,
    0.29, 0.41, 0.35, 0.98, 0.70, 1.64,
    40648177.18, 0.61, 13.64, 13.35, 1, 1, 0, 1,
]


@pytest.fixture
def mock_model_manager():
    with patch("api.main.model_manager") as mock_mm:
        mock_mm.model = MagicMock()
        mock_mm.model_name = "credit_card_propensity"
        mock_mm.model_version = "1"
        mock_mm.model_uri = "models:/credit_card_propensity/Production"
        mock_mm.load_time = 0.5
        mock_mm.predict.return_value = (0.72, 0.01)
        yield mock_mm


@pytest.fixture
def client(mock_model_manager):
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_contains_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data
        assert data["model_version"] == "1"


class TestPredictEndpoint:
    def test_predict_success(self, client):
        payload = {"features": SAMPLE_FEATURES}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_response_schema(self, client):
        payload = {"features": SAMPLE_FEATURES}
        data = client.post("/predict", json=payload).json()
        assert "propensity_score" in data
        assert "will_open_card" in data
        assert "model_name" in data
        assert "latency_ms" in data
        assert "timestamp" in data

    def test_predict_score_range(self, client):
        payload = {"features": SAMPLE_FEATURES}
        data = client.post("/predict", json=payload).json()
        assert 0.0 <= data["propensity_score"] <= 1.0

    def test_predict_with_feature_names(self, client):
        payload = {
            "features": SAMPLE_FEATURES,
            "feature_names": FEATURE_NAMES,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_with_customer_id(self, client):
        payload = {
            "features": SAMPLE_FEATURES,
            "customer_id": "CUST_001",
        }
        data = client.post("/predict", json=payload).json()
        assert data["customer_id"] == "CUST_001"

    def test_predict_wrong_feature_count(self, client):
        payload = {"features": [1.0, 2.0, 3.0]}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400

    def test_predict_empty_features(self, client):
        payload = {"features": []}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400

    def test_predict_threshold_logic(self, client):
        payload = {"features": SAMPLE_FEATURES}
        data = client.post("/predict", json=payload).json()
        score = data["propensity_score"]
        expected = score >= data["threshold"]
        assert data["will_open_card"] == expected


class TestModelInfoEndpoint:
    def test_model_info_success(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_model_info_fields(self, client):
        data = client.get("/model/info").json()
        assert "model_name" in data
        assert "model_version" in data
        assert "feature_names" in data
        assert len(data["feature_names"]) == 33

    def test_model_info_when_not_loaded(self):
        with patch("api.main.model_manager") as mock_mm:
            mock_mm.model = None
            from api.main import app
            c = TestClient(app)
            r = c.get("/model/info")
            assert r.status_code == 503


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert b"api_requests_total" in r.content or b"python_info" in r.content


class TestRootEndpoint:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "endpoints" in data
