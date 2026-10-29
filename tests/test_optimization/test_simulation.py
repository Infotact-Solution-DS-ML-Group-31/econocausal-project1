"""Unit tests for Monte Carlo Policy Confidence Simulator and REST endpoint."""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app
from src.api.mock_data import generate_mock_customers
from src.api.schemas import CustomerDataPoint, OptimizationRequest
from src.optimization.simulation import (
    MonteCarloSimulationResult,
    run_policy_monte_carlo,
)


class TestMonteCarloSimulation(unittest.TestCase):
    """Test suite for Monte Carlo policy confidence simulator and API endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_simulation_confidence_intervals_and_metrics(self):
        """Verify ci_lower_5pct <= mean_net_profit <= ci_upper_95pct and probability_of_positive_roi in [0, 1]."""
        mock_customers = generate_mock_customers(num_customers=50, seed=42)
        request = OptimizationRequest(budget=200.0, customers=mock_customers)

        result = run_policy_monte_carlo(
            request=request,
            ite_std_error=0.03,
            num_simulations=150,
            seed=42,
        )

        self.assertIsInstance(result, MonteCarloSimulationResult)
        self.assertEqual(result.num_simulations, 150)
        self.assertEqual(result.budget, 200.0)
        self.assertLessEqual(result.total_spend, 200.0)
        self.assertGreater(result.total_spend, 0.0)

        # Confidence interval bounds check
        self.assertLessEqual(result.ci_lower_5pct, result.mean_net_profit)
        self.assertLessEqual(result.mean_net_profit, result.ci_upper_95pct)
        self.assertGreater(result.std_net_profit, 0.0)

        # Probability bounds
        self.assertGreaterEqual(result.probability_of_positive_roi, 0.0)
        self.assertLessEqual(result.probability_of_positive_roi, 1.0)

    def test_zero_std_error_deterministic(self):
        """Verify that when ite_std_error == 0.0, std == 0.0 and ci_lower == ci_upper == mean."""
        mock_customers = generate_mock_customers(num_customers=40, seed=99)
        request = OptimizationRequest(budget=150.0, customers=mock_customers)

        result = run_policy_monte_carlo(
            request=request,
            ite_std_error=0.0,
            num_simulations=100,
            seed=42,
        )

        self.assertEqual(result.std_net_profit, 0.0)
        self.assertEqual(result.ci_lower_5pct, result.mean_net_profit)
        self.assertEqual(result.ci_upper_95pct, result.mean_net_profit)

    def test_frozen_dataclass_immutability(self):
        """Verify MonteCarloSimulationResult is an immutable frozen dataclass."""
        result = MonteCarloSimulationResult(
            num_simulations=100,
            budget=200.0,
            total_spend=150.0,
            mean_net_profit=50.0,
            std_net_profit=5.0,
            ci_lower_5pct=42.0,
            ci_upper_95pct=58.0,
            probability_of_positive_roi=0.95,
        )
        with self.assertRaises(FrozenInstanceError):
            result.mean_net_profit = 60.0  # type: ignore

    def test_input_validation(self):
        """Verify input validation rejects invalid std error and non-positive simulation counts."""
        mock_customers = generate_mock_customers(num_customers=10, seed=1)
        request = OptimizationRequest(budget=50.0, customers=mock_customers)

        # Negative ite_std_error
        with self.assertRaises(ValueError):
            run_policy_monte_carlo(request=request, ite_std_error=-0.05)

        # Invalid num_simulations
        with self.assertRaises(ValueError):
            run_policy_monte_carlo(request=request, num_simulations=0)

        with self.assertRaises(ValueError):
            run_policy_monte_carlo(request=request, num_simulations=-10)

    def test_sleeping_dogs_cohort_zero_allocation(self):
        """Verify cohort of all Sleeping Dogs results in zero spend and zero variance."""
        customers = [
            CustomerDataPoint(
                customer_id=f"cust_{i}",
                baseline_conversion_prob=0.30,
                predicted_ite=-0.05 * (i + 1),
                treatment_cost=10.0,
                customer_value=50.0,
            )
            for i in range(10)
        ]
        request = OptimizationRequest(budget=100.0, customers=customers)
        result = run_policy_monte_carlo(request=request, ite_std_error=0.03, num_simulations=50)

        self.assertEqual(result.total_spend, 0.0)
        self.assertEqual(result.mean_net_profit, 0.0)
        self.assertEqual(result.std_net_profit, 0.0)
        self.assertEqual(result.ci_lower_5pct, 0.0)
        self.assertEqual(result.ci_upper_95pct, 0.0)
        self.assertEqual(result.probability_of_positive_roi, 0.0)

    def test_post_simulate_confidence_endpoint_success(self):
        """Test POST /api/v1/simulate-confidence via TestClient asserting HTTP 200."""
        mock_customers = generate_mock_customers(num_customers=30, seed=77)
        payload = {
            "budget": 120.0,
            "objective": "maximize_profit",
            "min_spend_per_user": 0.0,
            "ite_std_error": 0.02,
            "num_simulations": 50,
            "seed": 42,
            "customers": [c.model_dump() for c in mock_customers],
        }

        response = self.client.post("/api/v1/simulate-confidence", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["num_simulations"], 50)
        self.assertEqual(data["budget"], 120.0)
        self.assertLessEqual(data["total_spend"], 120.0)
        self.assertLessEqual(data["ci_lower_5pct"], data["mean_net_profit"])
        self.assertLessEqual(data["mean_net_profit"], data["ci_upper_95pct"])
        self.assertGreaterEqual(data["probability_of_positive_roi"], 0.0)
        self.assertLessEqual(data["probability_of_positive_roi"], 1.0)

    def test_post_simulate_confidence_endpoint_validation_errors(self):
        """Verify API validation rejects out-of-bounds simulation parameters."""
        mock_customers = generate_mock_customers(num_customers=5, seed=1)

        # ite_std_error > 0.50
        payload = {
            "budget": 50.0,
            "ite_std_error": 0.60,
            "num_simulations": 50,
            "customers": [c.model_dump() for c in mock_customers],
        }
        res = self.client.post("/api/v1/simulate-confidence", json=payload)
        self.assertEqual(res.status_code, 422)

        # num_simulations < 10
        payload["ite_std_error"] = 0.03
        payload["num_simulations"] = 5
        res = self.client.post("/api/v1/simulate-confidence", json=payload)
        self.assertEqual(res.status_code, 422)

        # num_simulations > 2000
        payload["num_simulations"] = 2500
        res = self.client.post("/api/v1/simulate-confidence", json=payload)
        self.assertEqual(res.status_code, 422)

        # Duplicate customer_id
        dup_payload = {
            "budget": 50.0,
            "num_simulations": 50,
            "customers": [
                {
                    "customer_id": "cust_dup",
                    "baseline_conversion_prob": 0.2,
                    "predicted_ite": 0.1,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                },
                {
                    "customer_id": "cust_dup",
                    "baseline_conversion_prob": 0.3,
                    "predicted_ite": 0.15,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                },
            ],
        }
        res = self.client.post("/api/v1/simulate-confidence", json=dup_payload)
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()

