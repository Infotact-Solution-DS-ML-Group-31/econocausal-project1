"""API integration and stress tests for drift monitoring and allocation under high volume."""

import sys
import time
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.mock_data import generate_mock_customers, generate_mock_optimization_request


class TestDriftAndStressEndpoints(unittest.TestCase):
    """Integration and stress tests for /drift-check, error middleware, and /allocate."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.rng = np.random.default_rng(42)

    def test_drift_check_stable_distribution(self):
        """POST /api/v1/drift-check on stable distributions returns severity 'none'."""
        ref_samples = self.rng.normal(0.12, 0.02, 500).tolist()
        cur_samples = self.rng.normal(0.12, 0.02, 500).tolist()

        payload = {
            "reference_ites": ref_samples,
            "current_ites": cur_samples,
            "psi_threshold": 0.25,
            "ks_alpha": 0.05,
        }

        response = self.client.post("/api/v1/drift-check", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertFalse(data["drift_detected"])
        self.assertEqual(data["severity"], "none")
        self.assertLess(data["psi_score"], 0.25)
        self.assertEqual(data["reference_count"], 500)
        self.assertEqual(data["current_count"], 500)
        self.assertIn("No significant distribution drift detected", data["recommendation"])

    def test_drift_check_drifted_distribution(self):
        """POST /api/v1/drift-check on shifted distributions returns severity 'critical'."""
        ref_samples = self.rng.normal(0.05, 0.02, 500).tolist()
        cur_samples = self.rng.normal(0.25, 0.05, 500).tolist()

        payload = {
            "reference_ites": ref_samples,
            "current_ites": cur_samples,
            "psi_threshold": 0.25,
            "ks_alpha": 0.05,
        }

        response = self.client.post("/api/v1/drift-check", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data["drift_detected"])
        self.assertEqual(data["severity"], "critical")
        self.assertGreaterEqual(data["psi_score"], 0.25)
        self.assertLess(data["ks_pvalue"], 0.05)
        self.assertEqual(data["reference_count"], 500)
        self.assertEqual(data["current_count"], 500)
        self.assertIn("Critical distribution shift detected", data["recommendation"])
        self.assertIn("retrain EconML Double ML model", data["recommendation"])

    def test_drift_check_schema_validation_error(self):
        """POST /api/v1/drift-check with fewer than 5 data points returns HTTP 422."""
        payload = {
            "reference_ites": [0.1, 0.2],
            "current_ites": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        response = self.client.post("/api/v1/drift-check", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_global_exception_handler_catches_value_error(self):
        """Global exception handler returns 400 and ValidationError when ValueError is raised."""
        # Budget exceeding solver limit 1_000_000.0 passes Pydantic schema validation
        # but raises ValueError in validate_solver_inputs, caught by global exception handler
        req = generate_mock_optimization_request(num_customers=10, budget=2_000_000.0)

        response = self.client.post("/api/v1/allocate", json=req.model_dump())
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error_type"], "ValidationError")
        self.assertIn("exceeds maximum allowed limit", data["detail"])

    def test_high_volume_stress_test_1000_customers(self):
        """Stress test: 1,000 customers allocated under budget within 5.0 seconds."""
        budget = 2500.0
        req = generate_mock_optimization_request(num_customers=1000, budget=budget)
        payload = req.model_dump()

        start_time = time.perf_counter()
        response = self.client.post("/api/v1/allocate", json=payload)
        elapsed_sec = time.perf_counter() - start_time

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_customers_count"], 1000)
        self.assertEqual(len(data["allocations"]), 1000)
        self.assertLessEqual(data["total_spend"], budget)
        self.assertGreater(data["targeted_customers_count"], 0)
        self.assertLessEqual(data["budget_utilization_pct"], 100.0)
        self.assertLess(elapsed_sec, 5.0, f"Allocation took {elapsed_sec:.3f}s, exceeding 5.0s ceiling.")


if __name__ == "__main__":
    unittest.main()
