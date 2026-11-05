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