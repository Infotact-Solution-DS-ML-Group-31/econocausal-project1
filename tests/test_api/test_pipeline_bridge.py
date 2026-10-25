"""Unit test suite for pipeline bridge and ingestion-allocation endpoint."""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app
from src.api.pipeline_bridge import (
    allocate_discounts,
    build_request_from_causal_records,
    export_allocation_to_records,
)
from src.api.schemas import (
    CustomerAllocation,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResponse,
)


class TestPipelineBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_build_request_from_m2_raw_dicts(self):
        """Verify building an OptimizationRequest from M2-style raw dicts with varied aliases."""
        raw_records = [
            {"id": "u_1", "cate": 0.22, "prob": 0.15},
            {
                "customer_id": "u_2",
                "predicted_ite": 0.35,
                "baseline_conversion_prob": 0.10,
                "treatment_cost": 15.0,
                "customer_value": 75.0,
            },
            {"user_id": "u_3", "uplift": -0.05, "propensity": 0.20},
        ]

        req = build_request_from_causal_records(
            records=raw_records,
            budget=150.0,
            default_cost=10.0,
            default_value=50.0,
            objective=OptimizationObjective.MAXIMIZE_PROFIT,
        )

        self.assertIsInstance(req, OptimizationRequest)
        self.assertEqual(req.budget, 150.0)
        self.assertEqual(req.objective, OptimizationObjective.MAXIMIZE_PROFIT)
        self.assertEqual(len(req.customers), 3)

        # Record 1 (aliases: id, cate, prob; default cost & value applied)
        c1 = req.customers[0]
        self.assertEqual(c1.customer_id, "u_1")
        self.assertEqual(c1.predicted_ite, 0.22)
        self.assertEqual(c1.baseline_conversion_prob, 0.15)
        self.assertEqual(c1.treatment_cost, 10.0)
        self.assertEqual(c1.customer_value, 50.0)

        # Record 2 (explicit fields)
        c2 = req.customers[1]
        self.assertEqual(c2.customer_id, "u_2")
        self.assertEqual(c2.predicted_ite, 0.35)
        self.assertEqual(c2.baseline_conversion_prob, 0.10)
        self.assertEqual(c2.treatment_cost, 15.0)
        self.assertEqual(c2.customer_value, 75.0)

        # Record 3 (aliases: user_id, uplift, propensity; sleeping dog)
        c3 = req.customers[2]
        self.assertEqual(c3.customer_id, "u_3")
        self.assertEqual(c3.predicted_ite, -0.05)
        self.assertEqual(c3.baseline_conversion_prob, 0.20)
        self.assertEqual(c3.treatment_cost, 10.0)
        self.assertEqual(c3.customer_value, 50.0)

    def test_probability_clamping_and_validation(self):
        """Verify baseline conversion probabilities are clamped to [0.0, 1.0]."""
        raw_records = [
            {"id": "u_low", "cate": 0.10, "prob": -0.45},
            {"id": "u_high", "cate": 0.15, "prob": 1.75},
        ]
        req = build_request_from_causal_records(records=raw_records, budget=50.0)

        self.assertEqual(req.customers[0].baseline_conversion_prob, 0.0)
        self.assertEqual(req.customers[1].baseline_conversion_prob, 1.0)

    def test_build_request_invalid_inputs(self):
        """Empty records or non-positive budget must raise ValueError."""
        with self.assertRaises(ValueError):
            build_request_from_causal_records(records=[], budget=100.0)

        with self.assertRaises(ValueError):
            build_request_from_causal_records(
                records=[{"id": "u_1", "cate": 0.1}], budget=0.0
            )

        with self.assertRaises(ValueError):
            build_request_from_causal_records(
                records=[{"id": "u_1", "cate": 0.1}], budget=-25.0
            )

    def test_export_allocation_to_records_and_roi_calculation(self):
        """Verify export_allocation_to_records flattens response and calculates ROI % correctly."""
        allocations = [
            CustomerAllocation(
                customer_id="cust_1",
                assigned_discount=10.0,
                is_targeted=True,
                expected_incremental_revenue=30.0,
                expected_net_profit=20.0,
            ),
            CustomerAllocation(
                customer_id="cust_2",
                assigned_discount=20.0,
                is_targeted=True,
                expected_incremental_revenue=35.0,
                expected_net_profit=15.0,
            ),
            CustomerAllocation(
                customer_id="cust_3",
                assigned_discount=0.0,
                is_targeted=False,
                expected_incremental_revenue=0.0,
                expected_net_profit=0.0,
            ),
        ]
        mock_response = OptimizationResponse(
            success=True,
            status_message="Optimal solution found",
            total_budget=100.0,
            total_spend=30.0,
            budget_utilization_pct=30.0,
            total_expected_incremental_revenue=65.0,
            total_expected_net_profit=35.0,
            targeted_customers_count=2,
            total_customers_count=3,
            allocations=allocations,
        )

        flat_records = export_allocation_to_records(mock_response)
        self.assertEqual(len(flat_records), 3)

        expected_keys = {
            "customer_id",
            "is_targeted",
            "assigned_discount",
            "expected_incremental_revenue",
            "expected_net_profit",
            "roi_pct",
        }
        for rec in flat_records:
            self.assertEqual(set(rec.keys()), expected_keys)

        # cust_1: (20.0 / 10.0) * 100 = 200.0%
        self.assertEqual(flat_records[0]["customer_id"], "cust_1")
        self.assertTrue(flat_records[0]["is_targeted"])
        self.assertEqual(flat_records[0]["assigned_discount"], 10.0)
        self.assertEqual(flat_records[0]["roi_pct"], 200.0)

        # cust_2: (15.0 / 20.0) * 100 = 75.0%
        self.assertEqual(flat_records[1]["customer_id"], "cust_2")
        self.assertTrue(flat_records[1]["is_targeted"])
        self.assertEqual(flat_records[1]["assigned_discount"], 20.0)
        self.assertEqual(flat_records[1]["roi_pct"], 75.0)

        # cust_3: untargeted -> roi_pct = 0.0
        self.assertEqual(flat_records[2]["customer_id"], "cust_3")
        self.assertFalse(flat_records[2]["is_targeted"])
        self.assertEqual(flat_records[2]["assigned_discount"], 0.0)
        self.assertEqual(flat_records[2]["roi_pct"], 0.0)

    def test_post_ingest_and_allocate_endpoint_success(self):
        """POST /api/v1/ingest-and-allocate accepts raw M2 records, runs optimization, returns 200."""
        payload = {
            "budget": 40.0,
            "default_cost": 10.0,
            "default_value": 50.0,
            "objective": "maximize_profit",
            "records": [
                {"id": "user_alpha", "cate": 0.30, "prob": 0.12},   # Profit = 0.30*50 - 10 = +5.0 (Persuadable)
                {"id": "user_beta", "cate": 0.40, "prob": 0.15},    # Profit = 0.40*50 - 10 = +10.0 (Persuadable)
                {"id": "user_gamma", "cate": -0.10, "prob": 0.08},  # Sleeping dog (negative uplift)
                {"id": "user_delta", "cate": 0.05, "prob": 0.20},   # Profit = 0.05*50 - 10 = -7.5 (Unprofitable)
            ],
        }

        response = self.client.post("/api/v1/ingest-and-allocate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_budget"], 40.0)
        self.assertLessEqual(data["total_spend"], 40.0)
        self.assertEqual(data["total_customers_count"], 4)
        self.assertEqual(data["targeted_customers_count"], 2)

        # Check customer allocations
        alloc_map = {a["customer_id"]: a for a in data["allocations"]}
        self.assertTrue(alloc_map["user_alpha"]["is_targeted"])
        self.assertEqual(alloc_map["user_alpha"]["assigned_discount"], 10.0)

        self.assertTrue(alloc_map["user_beta"]["is_targeted"])
        self.assertEqual(alloc_map["user_beta"]["assigned_discount"], 10.0)

        # Sleeping dog & unprofitable must be untargeted
        self.assertFalse(alloc_map["user_gamma"]["is_targeted"])
        self.assertEqual(alloc_map["user_gamma"]["assigned_discount"], 0.0)

        self.assertFalse(alloc_map["user_delta"]["is_targeted"])
        self.assertEqual(alloc_map["user_delta"]["assigned_discount"], 0.0)

        # Verify export function works seamlessly with the endpoint's response model
        parsed_response = OptimizationResponse(**data)
        flat = export_allocation_to_records(parsed_response)
        self.assertEqual(len(flat), 4)

    def test_post_ingest_and_allocate_validation_errors(self):
        """Invalid requests (empty records or non-positive budget) return 422."""
        # Empty records
        res1 = self.client.post(
            "/api/v1/ingest-and-allocate",
            json={"budget": 50.0, "records": []},
        )
        self.assertEqual(res1.status_code, 422)

        # Non-positive budget
        res2 = self.client.post(
            "/api/v1/ingest-and-allocate",
            json={
                "budget": 0.0,
                "records": [{"id": "u_1", "cate": 0.2, "prob": 0.1}],
            },
        )
        self.assertEqual(res2.status_code, 422)


if __name__ == "__main__":
    unittest.main()
