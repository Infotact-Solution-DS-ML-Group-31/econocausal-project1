from fastapi.testclient import TestClient

from econocausal.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_causal_readiness_reports_not_ready():
    response = client.get("/causal/readiness")

    assert response.status_code == 200

    data = response.json()
    assert data["causal_estimation_ready"] is False
    assert data["status"] == "not_ready"


def test_causal_validation_accepts_valid_structure():
    response = client.post(
        "/causal/validate",
        json={
            "records": [
                {
                    "customer_id": "C1",
                    "product_id": "P1",
                    "treatment": 1,
                    "outcome": 1,
                    "assignment_date": "2026-01-01",
                    "age": 25,
                },
                {
                    "customer_id": "C2",
                    "product_id": "P1",
                    "treatment": 0,
                    "outcome": 0,
                    "assignment_date": "2026-01-02",
                    "age": 30,
                },
            ],
            "covariates": ["age"],
        },
    )

    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert response.json()["causal_identification_established"] is False


def test_causal_validation_rejects_missing_columns():
    response = client.post(
        "/causal/validate",
        json={
            "records": [
                {"customer_id": "C1", "treatment": 1}
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["errors"]


def test_optimizer_endpoint_allocates_within_budget():
    response = client.post(
        "/optimization/allocate",
        json={
            "budget": 9,
            "offers": [
                {
                    "customer_id": "C1",
                    "offer_id": "A1",
                    "cost": 5,
                    "expected_incremental_value": 10,
                },
                {
                    "customer_id": "C1",
                    "offer_id": "A2",
                    "cost": 6,
                    "expected_incremental_value": 12,
                },
                {
                    "customer_id": "C2",
                    "offer_id": "B1",
                    "cost": 3,
                    "expected_incremental_value": 5,
                },
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["total_cost"] <= data["budget"]
    assert data["total_expected_incremental_value"] == 17
    assert data["selected_count"] == 2
    assert data["causal_estimates_generated"] is False


def test_optimizer_rejects_invalid_budget():
    response = client.post(
        "/optimization/allocate",
        json={
            "budget": -1,
            "offers": [
                {
                    "customer_id": "C1",
                    "offer_id": "A1",
                    "cost": 5,
                    "expected_incremental_value": 10,
                }
            ],
        },
    )

    assert response.status_code == 422


def test_optimizer_rejects_duplicate_offer_ids():
    response = client.post(
        "/optimization/allocate",
        json={
            "budget": 20,
            "offers": [
                {
                    "customer_id": "C1",
                    "offer_id": "A1",
                    "cost": 5,
                    "expected_incremental_value": 10,
                },
                {
                    "customer_id": "C2",
                    "offer_id": "A1",
                    "cost": 6,
                    "expected_incremental_value": 12,
                },
            ],
        },
    )

    assert response.status_code == 400

def test_uplift_endpoint_returns_metrics():
    from fastapi.testclient import TestClient
    from econocausal.api.main import app

    client = TestClient(app)

    response = client.post(
        "/causal/uplift",
        json={
            "treatment": [1, 1, 1, 1, 0, 0, 0, 0],
            "outcome": [1, 1, 0, 1, 0, 0, 1, 0],
            "predicted_uplift": [
                0.9, 0.8, 0.7, 0.6,
                0.4, 0.3, 0.2, 0.1
            ],
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "auuc" in result
    assert "qini_coefficient" in result
    assert result["n_observations"] == 8
    assert result["causal_identification_established"] is False


def test_uplift_endpoint_rejects_invalid_treatment():
    from fastapi.testclient import TestClient
    from econocausal.api.main import app

    client = TestClient(app)

    response = client.post(
        "/causal/uplift",
        json={
            "treatment": [0.5, 1, 1, 0],
            "outcome": [1, 1, 0, 0],
            "predicted_uplift": [0.9, 0.8, 0.7, 0.6],
        },
    )

    assert response.status_code == 400


def test_drift_endpoint_detects_numeric_and_categorical_drift():
    response = client.post(
        "/monitoring/drift",
        json={
            "reference_records": [
                {"age": 20, "segment": "A"},
                {"age": 25, "segment": "A"},
                {"age": 30, "segment": "A"},
                {"age": 35, "segment": "B"},
                {"age": 40, "segment": "B"},
                {"age": 45, "segment": "B"},
            ],
            "current_records": [
                {"age": 55, "segment": "B"},
                {"age": 60, "segment": "B"},
                {"age": 65, "segment": "B"},
                {"age": 70, "segment": "B"},
                {"age": 75, "segment": "A"},
                {"age": 80, "segment": "A"},
            ],
            "features": ["age", "segment"],
            "bins": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ["warning", "drift_detected"]
    assert data["features_checked"] == 2
    assert len(data["results"]) == 2

    feature_types = {
        result["feature"]: result["feature_type"]
        for result in data["results"]
    }

    assert feature_types["age"] == "numeric"
    assert feature_types["segment"] == "categorical"


def test_drift_endpoint_rejects_empty_reference_records():
    response = client.post(
        "/monitoring/drift",
        json={
            "reference_records": [],
            "current_records": [{"age": 30}],
        },
    )

    assert response.status_code == 422


def test_drift_endpoint_rejects_empty_current_records():
    response = client.post(
        "/monitoring/drift",
        json={
            "reference_records": [{"age": 30}],
            "current_records": [],
        },
    )

    assert response.status_code == 422


def test_drift_endpoint_rejects_missing_features():
    response = client.post(
        "/monitoring/drift",
        json={
            "reference_records": [{"age": 20}],
            "current_records": [{"income": 5000}],
            "features": ["age"],
        },
    )

    assert response.status_code == 400
    assert "Missing features" in response.json()["detail"]


def test_psm_endpoint_returns_matching_and_balance_metrics():
    response = client.post(
        "/causal/psm",
        json={
            "records": [
                {"treatment": 1, "outcome": 1, "age": 25},
                {"treatment": 1, "outcome": 0, "age": 30},
                {"treatment": 1, "outcome": 1, "age": 35},
                {"treatment": 1, "outcome": 0, "age": 40},
                {"treatment": 0, "outcome": 0, "age": 26},
                {"treatment": 0, "outcome": 0, "age": 31},
                {"treatment": 0, "outcome": 0, "age": 36},
                {"treatment": 0, "outcome": 0, "age": 41},
            ],
            "covariates": ["age"],
            "replacement": False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["n_observations"] == 8
    assert data["n_treated"] == 4
    assert data["n_control"] == 4
    assert "common_support" in data
    assert "matching" in data
    assert "att" in data
    assert "smd_before" in data
    assert "smd_after" in data
    assert data["matching"]["n_matched"] > 0
    assert data["causal_identification_established"] is False


def test_psm_endpoint_rejects_missing_covariate():
    response = client.post(
        "/causal/psm",
        json={
            "records": [
                {"treatment": 1, "outcome": 1, "age": 25},
                {"treatment": 0, "outcome": 0, "age": 30},
            ],
            "covariates": ["income"],
        },
    )

    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]
