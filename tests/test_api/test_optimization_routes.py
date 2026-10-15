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


if __name__ == "__main__":
    unittest.main()
