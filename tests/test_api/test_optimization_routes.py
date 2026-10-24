import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.mock_data import generate_mock_customers, generate_mock_optimization_request
from src.api.schemas import OptimizationObjective


class TestOptimizationRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_allocate_endpoint_success(self):
        """POST /api/v1/allocate returns 200, valid schema, and respects budget."""
        budget = 250.0
        req = generate_mock_optimization_request(num_customers=50, budget=budget)
        payload = req.model_dump()

        response = self.client.post("/api/v1/allocate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_budget"], budget)
        self.assertLessEqual(data["total_spend"], budget)
        self.assertGreaterEqual(data["total_spend"], 0.0)
        self.assertLessEqual(data["budget_utilization_pct"], 100.0)
        self.assertEqual(data["total_customers_count"], 50)
        self.assertEqual(len(data["allocations"]), 50)
        self.assertEqual(data["metadata"]["solver"], "SciPyConstrainedSolver (HiGHS)")
        self.assertEqual(data["metadata"]["objective"], "maximize_profit")

        # Verify allocation schema structure
        first_alloc = data["allocations"][0]
        self.assertIn("customer_id", first_alloc)
        self.assertIn("assigned_discount", first_alloc)
        self.assertIn("is_targeted", first_alloc)
        self.assertIn("expected_incremental_revenue", first_alloc)
        self.assertIn("expected_net_profit", first_alloc)

    def test_allocate_different_objectives(self):
        """Verify allocate endpoint supports maximize_revenue and maximize_conversions."""
        for obj in [OptimizationObjective.MAXIMIZE_REVENUE, OptimizationObjective.MAXIMIZE_CONVERSIONS]:
            req = generate_mock_optimization_request(
                num_customers=30, budget=150.0, objective=obj
            )
            response = self.client.post("/api/v1/allocate", json=req.model_dump())
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["metadata"]["objective"], obj.value)
            self.assertLessEqual(data["total_spend"], 150.0)

    def test_compare_policies_endpoint(self):
        """POST /api/v1/compare-policies benchmarks causal optimizer against blanket baseline."""
        budget = 200.0
        req = generate_mock_optimization_request(num_customers=50, budget=budget)
        payload = req.model_dump()

        response = self.client.post("/api/v1/compare-policies", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("budget", data)
        self.assertIn("causal_scipy_policy", data)
        self.assertIn("blanket_baseline_policy", data)
        self.assertIn("net_profit_improvement", data)
        self.assertIn("executive_summary", data)

        self.assertEqual(data["budget"], budget)

        scipy_policy = data["causal_scipy_policy"]
        blanket_policy = data["blanket_baseline_policy"]

        self.assertLessEqual(scipy_policy["total_spend"], budget)
        self.assertLessEqual(blanket_policy["total_spend"], budget)

        # Causal optimizer should produce superior net profit
        self.assertGreaterEqual(
            scipy_policy["total_expected_net_profit"],
            blanket_policy["total_expected_net_profit"],
        )
        self.assertIn("Causal optimization improves net profit", data["executive_summary"])

    def test_allocate_budget_below_minimum_cost(self):
        """Budget below minimum treatment cost returns zero allocations without failure."""
        customers = generate_mock_customers(
            num_customers=10, treatment_cost=20.0, customer_value=200.0
        )
        payload = {
            "budget": 10.0,
            "objective": "maximize_profit",
            "min_spend_per_user": 0.0,
            "customers": [c.model_dump() for c in customers],
        }

        response = self.client.post("/api/v1/allocate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_spend"], 0.0)
        self.assertEqual(data["targeted_customers_count"], 0)
        self.assertIn("Budget below minimum treatment cost; zero allocations made", data["status_message"])

    def test_allocate_validation_errors(self):
        """Invalid requests (non-positive budget, empty customer list) should return 422."""
        # Non-positive budget
        res1 = self.client.post(
            "/api/v1/allocate",
            json={"budget": 0.0, "customers": [c.model_dump() for c in generate_mock_customers(5)]},
        )
        self.assertEqual(res1.status_code, 422)

        # Empty customers
        res2 = self.client.post(
            "/api/v1/allocate",
            json={"budget": 100.0, "customers": []},
        )
        self.assertEqual(res2.status_code, 422)

    def test_allocate_multi_tier_endpoint_success(self):
        """POST /api/v1/allocate-multi-tier returns 200, respects budget and mutual exclusivity."""
        budget = 40.0
        payload = {
            "budget": budget,
            "tier_costs": [10.0, 20.0],
            "objective": "maximize_profit",
            "customers": [
                {
                    "customer_id": "cust_1",
                    "baseline_conversion_prob": 0.10,
                    "tier_ites": [0.30, 0.35],
                    "customer_value": 100.0,
                },
                {
                    "customer_id": "cust_2",
                    "baseline_conversion_prob": 0.15,
                    "tier_ites": [0.20, 0.45],
                    "customer_value": 100.0,
                },
                {
                    "customer_id": "cust_3",
                    "baseline_conversion_prob": 0.05,
                    "tier_ites": [-0.10, -0.15],
                    "customer_value": 100.0,
                },
                {
                    "customer_id": "cust_4",
                    "baseline_conversion_prob": 0.20,
                    "tier_ites": [0.05, 0.08],
                    "customer_value": 100.0,
                },
                {
                    "customer_id": "cust_5",
                    "baseline_conversion_prob": 0.12,
                    "tier_ites": [0.25, 0.30],
                    "customer_value": 100.0,
                },
            ],
        }

        response = self.client.post("/api/v1/allocate-multi-tier", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_budget"], budget)
        self.assertLessEqual(data["total_spend"], budget)
        self.assertEqual(data["total_spend"], 40.0)
        self.assertEqual(data["total_customers_count"], 5)
        self.assertEqual(len(data["allocations"]), 5)
        self.assertEqual(data["targeted_customers_count"], 3)
        self.assertEqual(data["metadata"]["objective"], "maximize_profit")

        # Verify mutual exclusivity: Each customer receives at most one tier
        allocations_by_id = {a["customer_id"]: a for a in data["allocations"]}
        
        # cust_1 prefers $10 tier (profit 20 > 15)
        self.assertTrue(allocations_by_id["cust_1"]["is_targeted"])
        self.assertEqual(allocations_by_id["cust_1"]["assigned_discount"], 10.0)
        self.assertEqual(allocations_by_id["cust_1"]["expected_incremental_revenue"], 30.0)
        self.assertEqual(allocations_by_id["cust_1"]["expected_net_profit"], 20.0)

        # cust_2 prefers $20 tier (profit 25 > 10)
        self.assertTrue(allocations_by_id["cust_2"]["is_targeted"])
        self.assertEqual(allocations_by_id["cust_2"]["assigned_discount"], 20.0)
        self.assertEqual(allocations_by_id["cust_2"]["expected_incremental_revenue"], 45.0)
        self.assertEqual(allocations_by_id["cust_2"]["expected_net_profit"], 25.0)

        # cust_3 is a sleeping dog (negative uplift) -> untargeted
        self.assertFalse(allocations_by_id["cust_3"]["is_targeted"])
        self.assertEqual(allocations_by_id["cust_3"]["assigned_discount"], 0.0)

        # cust_4 is unprofitable -> untargeted
        self.assertFalse(allocations_by_id["cust_4"]["is_targeted"])
        self.assertEqual(allocations_by_id["cust_4"]["assigned_discount"], 0.0)

        # cust_5 prefers $10 tier (profit 15 > 10)
        self.assertTrue(allocations_by_id["cust_5"]["is_targeted"])
        self.assertEqual(allocations_by_id["cust_5"]["assigned_discount"], 10.0)
        self.assertEqual(allocations_by_id["cust_5"]["expected_incremental_revenue"], 25.0)
        self.assertEqual(allocations_by_id["cust_5"]["expected_net_profit"], 15.0)

        # Assert all assigned discounts are exclusively in {0.0, 10.0, 20.0}
        for alloc in data["allocations"]:
            self.assertIn(alloc["assigned_discount"], [0.0, 10.0, 20.0])
            if alloc["is_targeted"]:
                self.assertGreater(alloc["assigned_discount"], 0.0)
            else:
                self.assertEqual(alloc["assigned_discount"], 0.0)

    def test_allocate_multi_tier_validation_errors(self):
        """Invalid requests (mismatched tier lengths, duplicate customer IDs, non-positive budget) return 422."""
        # 1. Mismatched tier_ites length vs tier_costs length
        mismatched_payload = {
            "budget": 50.0,
            "tier_costs": [10.0, 20.0],
            "customers": [
                {
                    "customer_id": "cust_mismatch",
                    "baseline_conversion_prob": 0.1,
                    "tier_ites": [0.15],  # 1 tier given, expected 2
                    "customer_value": 50.0,
                }
            ],
        }
        res_mismatch = self.client.post("/api/v1/allocate-multi-tier", json=mismatched_payload)
        self.assertEqual(res_mismatch.status_code, 422)

        # 2. Duplicate customer IDs
        dup_payload = {
            "budget": 50.0,
            "tier_costs": [10.0, 20.0],
            "customers": [
                {
                    "customer_id": "cust_dup",
                    "baseline_conversion_prob": 0.1,
                    "tier_ites": [0.15, 0.25],
                    "customer_value": 50.0,
                },
                {
                    "customer_id": "cust_dup",
                    "baseline_conversion_prob": 0.2,
                    "tier_ites": [0.10, 0.20],
                    "customer_value": 50.0,
                },
            ],
        }
        res_dup = self.client.post("/api/v1/allocate-multi-tier", json=dup_payload)
        self.assertEqual(res_dup.status_code, 422)

        # 3. Non-positive budget
        res_zero_budget = self.client.post(
            "/api/v1/allocate-multi-tier",
            json={
                "budget": 0.0,
                "tier_costs": [10.0, 20.0],
                "customers": [
                    {
                        "customer_id": "c1",
                        "baseline_conversion_prob": 0.1,
                        "tier_ites": [0.1, 0.2],
                    }
                ],
            },
        )
        self.assertEqual(res_zero_budget.status_code, 422)

    def test_allocate_multi_tier_budget_below_min_cost(self):
        """Budget below minimum tier cost returns zero allocations without failure."""
        payload = {
            "budget": 5.0,  # Below min tier cost of 10.0
            "tier_costs": [10.0, 20.0],
            "objective": "maximize_profit",
            "customers": [
                {
                    "customer_id": "cust_1",
                    "baseline_conversion_prob": 0.10,
                    "tier_ites": [0.30, 0.35],
                    "customer_value": 100.0,
                }
            ],
        }
        response = self.client.post("/api/v1/allocate-multi-tier", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_spend"], 0.0)
        self.assertEqual(data["targeted_customers_count"], 0)
        self.assertIn("Budget below minimum treatment cost", data["status_message"])
        self.assertEqual(data["allocations"][0]["assigned_discount"], 0.0)
        self.assertFalse(data["allocations"][0]["is_targeted"])


if __name__ == "__main__":
    unittest.main()
