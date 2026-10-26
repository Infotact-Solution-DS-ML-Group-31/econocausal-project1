"""Unit tests for causal uplift curve, AUUC, and Qini evaluation metrics."""

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
from src.api.schemas import CustomerDataPoint, UpliftCurveRequest
from src.optimization.uplift_metrics import (
    UpliftCurveReport,
    UpliftDecilePoint,
    compute_uplift_curve,
)


class TestUpliftMetrics(unittest.TestCase):
    """Test suite covering causal uplift deciles, AUUC score, Qini coefficient, and API route."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_mixed_cohort_yields_positive_qini(self):
        """Verify sorting by causal uplift yields qini_coefficient > 0.0 on mixed customer cohorts."""
        # 10 customers: 5 strong Persuadables + 5 Sleeping Dogs
        treatment_effects = np.array([0.30, 0.25, 0.20, 0.15, 0.10, -0.05, -0.10, -0.15, -0.20, -0.25])
        treatment_costs = np.array([10.0] * 10)
        customer_values = np.array([100.0] * 10)

        report = compute_uplift_curve(
            treatment_effects=treatment_effects,
            treatment_costs=treatment_costs,
            customer_values=customer_values,
            num_bins=10,
        )

        self.assertIsInstance(report, UpliftCurveReport)
        self.assertEqual(len(report.deciles), 10)

        # Causal sorting prioritizes positive incremental revenue first
        self.assertGreater(report.qini_coefficient, 0.0)
        self.assertGreater(report.auuc_score, 0.0)

        # Confirm intermediate cumulative revenue points exceed random baseline
        for point in report.deciles[:5]:
            self.assertGreater(
                point.cumulative_incremental_revenue,
                point.random_baseline_revenue,
                f"At {point.percentile}%, causal curve should beat random baseline.",
            )

        # At 100% rollout, random baseline and cumulative revenue must converge
        last_point = report.deciles[-1]
        self.assertEqual(last_point.percentile, 100.0)
        self.assertEqual(last_point.customer_count, 10)
        self.assertAlmostEqual(
            last_point.cumulative_incremental_revenue,
            last_point.random_baseline_revenue,
            places=4,
        )

    def test_sleeping_dogs_max_profit_percentile_less_than_100(self):
        """Verify max_profit_percentile < 100.0 when Sleeping Dogs are present."""
        # Persuadables create net-positive profit; Sleeping Dogs create negative profit
        # Customer 0..4: ITE * 100 - 10 = +20, +15, +10, +5, 0
        # Customer 5..9: ITE * 100 - 10 = -15, -20, -25, -30, -35
        treatment_effects = np.array([0.30, 0.25, 0.20, 0.15, 0.10, -0.05, -0.10, -0.15, -0.20, -0.25])
        treatment_costs = np.array([10.0] * 10)
        customer_values = np.array([100.0] * 10)

        report = compute_uplift_curve(
            treatment_effects=treatment_effects,
            treatment_costs=treatment_costs,
            customer_values=customer_values,
            num_bins=10,
        )

        # Profit should peak before 100% (stopping avoids negative-ROI sleeping dogs)
        self.assertLess(report.max_profit_percentile, 100.0)
        self.assertIn(report.max_profit_percentile, [40.0, 50.0])

        profit_at_max = next(
            d.cumulative_net_profit
            for d in report.deciles
            if d.percentile == report.max_profit_percentile
        )
        profit_at_100 = report.deciles[-1].cumulative_net_profit

        self.assertGreater(
            profit_at_max,
            profit_at_100,
            "Stopping rollout at max_profit_percentile must strictly outperform full 100% rollout.",
        )

    def test_all_persuadables_max_profit_at_100(self):
        """Verify max_profit_percentile == 100.0 when all customers have positive ROI."""
        treatment_effects = np.array([0.40, 0.35, 0.30, 0.25, 0.20])
        treatment_costs = np.array([5.0] * 5)
        customer_values = np.array([50.0] * 5)

        report = compute_uplift_curve(
            treatment_effects=treatment_effects,
            treatment_costs=treatment_costs,
            customer_values=customer_values,
            num_bins=5,
        )

        self.assertEqual(report.max_profit_percentile, 100.0)
        self.assertGreater(report.qini_coefficient, 0.0)
        self.assertEqual(len(report.deciles), 5)

    def test_frozen_dataclass_immutability(self):
        """Verify UpliftDecilePoint and UpliftCurveReport are immutable."""
        point = UpliftDecilePoint(
            percentile=10.0,
            customer_count=5,
            cumulative_spend=50.0,
            cumulative_incremental_conversions=1.2,
            cumulative_incremental_revenue=60.0,
            cumulative_net_profit=10.0,
            random_baseline_revenue=6.0,
        )
        with self.assertRaises(FrozenInstanceError):
            point.percentile = 20.0

        report = UpliftCurveReport(
            deciles=[point],
            auuc_score=25.0,
            qini_coefficient=0.45,
            max_profit_percentile=50.0,
        )
        with self.assertRaises(FrozenInstanceError):
            report.auuc_score = 30.0

    def test_compute_uplift_curve_input_validation(self):
        """Verify input validation handles invalid bins, dimensions, NaNs, and negative values."""
        te = np.array([0.1, 0.2, 0.3])
        tc = np.array([10.0, 10.0, 10.0])
        cv = np.array([50.0, 50.0, 50.0])

        # num_bins < 2
        with self.assertRaises(ValueError):
            compute_uplift_curve(te, tc, cv, num_bins=1)

        # 2D treatment_effects
        with self.assertRaises(ValueError):
            compute_uplift_curve(te.reshape(1, -1), tc, cv)

        # customer_values is None
        with self.assertRaises(ValueError):
            compute_uplift_curve(te, tc, None)

        # Empty array
        with self.assertRaises(ValueError):
            compute_uplift_curve(np.array([]), np.array([]), np.array([]))

        # NaN values
        with self.assertRaises(ValueError):
            compute_uplift_curve(np.array([np.nan, 0.2]), np.array([10.0, 10.0]), np.array([50.0, 50.0]))

        # Negative costs
        with self.assertRaises(ValueError):
            compute_uplift_curve(te, np.array([-10.0, 10.0, 10.0]), cv)

        # Non-positive customer values
        with self.assertRaises(ValueError):
            compute_uplift_curve(te, tc, np.array([0.0, 50.0, 50.0]))

        # Mismatched lengths
        with self.assertRaises(ValueError):
            compute_uplift_curve(te, tc[:2], cv)

    def test_custom_num_bins(self):
        """Verify custom num_bins splits population into corresponding buckets."""
        te = np.linspace(-0.2, 0.4, 20)
        tc = np.full(20, 10.0)
        cv = np.full(20, 50.0)

        for num_bins in [4, 5, 20]:
            report = compute_uplift_curve(te, tc, cv, num_bins=num_bins)
            self.assertEqual(len(report.deciles), num_bins)
            self.assertEqual(report.deciles[-1].percentile, 100.0)
            self.assertEqual(report.deciles[-1].customer_count, 20)
            self.assertEqual(report.deciles[0].percentile, round(100.0 / num_bins, 2))

    def test_post_uplift_curve_endpoint_success(self):
        """Test POST /api/v1/uplift-curve via TestClient asserting HTTP 200 and valid decile coordinates."""
        # 10 customers: 5 persuadables, 5 sleeping dogs
        customers = [
            CustomerDataPoint(
                customer_id=f"c_{i}",
                baseline_conversion_prob=0.10,
                predicted_ite=0.30 - (i * 0.06),  # 0.30, 0.24, 0.18, 0.12, 0.06, 0.0, -0.06, -0.12, -0.18, -0.24
                treatment_cost=10.0,
                customer_value=100.0,
            )
            for i in range(10)
        ]

        request_payload = {
            "customers": [c.model_dump() for c in customers],
            "num_bins": 10,
        }

        response = self.client.post("/api/v1/uplift-curve", json=request_payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("deciles", data)
        self.assertIn("auuc_score", data)
        self.assertIn("qini_coefficient", data)
        self.assertIn("max_profit_percentile", data)
        self.assertIn("total_customers", data)

        self.assertEqual(data["total_customers"], 10)
        self.assertEqual(len(data["deciles"]), 10)
        self.assertGreater(data["qini_coefficient"], 0.0)
        self.assertLess(data["max_profit_percentile"], 100.0)

        # Verify decile coordinates and fields
        percentiles = [d["percentile"] for d in data["deciles"]]
        expected_percentiles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.assertEqual(percentiles, expected_percentiles)

        first_decile = data["deciles"][0]
        self.assertEqual(first_decile["percentile"], 10.0)
        self.assertEqual(first_decile["customer_count"], 1)
        self.assertIn("cumulative_spend", first_decile)
        self.assertIn("cumulative_incremental_conversions", first_decile)
        self.assertIn("cumulative_incremental_revenue", first_decile)
        self.assertIn("cumulative_net_profit", first_decile)
        self.assertIn("random_baseline_revenue", first_decile)

        # Last decile includes all customers
        last_decile = data["deciles"][-1]
        self.assertEqual(last_decile["percentile"], 100.0)
        self.assertEqual(last_decile["customer_count"], 10)
        self.assertAlmostEqual(
            last_decile["cumulative_incremental_revenue"],
            last_decile["random_baseline_revenue"],
            places=3,
        )

    def test_post_uplift_curve_validation_errors(self):
        """Verify API schema validation rejects duplicate IDs, empty lists, and invalid num_bins."""
        # Duplicate customer IDs
        dup_payload = {
            "customers": [
                {
                    "customer_id": "dup_1",
                    "baseline_conversion_prob": 0.1,
                    "predicted_ite": 0.2,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                },
                {
                    "customer_id": "dup_1",
                    "baseline_conversion_prob": 0.2,
                    "predicted_ite": 0.1,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                },
            ],
            "num_bins": 10,
        }
        res_dup = self.client.post("/api/v1/uplift-curve", json=dup_payload)
        self.assertEqual(res_dup.status_code, 422)

        # Empty customers list
        empty_payload = {"customers": [], "num_bins": 10}
        res_empty = self.client.post("/api/v1/uplift-curve", json=empty_payload)
        self.assertEqual(res_empty.status_code, 422)

        # num_bins < 2
        bad_bins_payload = {
            "customers": [
                {
                    "customer_id": "c_1",
                    "baseline_conversion_prob": 0.1,
                    "predicted_ite": 0.2,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                }
            ],
            "num_bins": 1,
        }
        res_bad_bins = self.client.post("/api/v1/uplift-curve", json=bad_bins_payload)
        self.assertEqual(res_bad_bins.status_code, 422)

        # num_bins > 50
        too_many_bins = {
            "customers": [
                {
                    "customer_id": "c_1",
                    "baseline_conversion_prob": 0.1,
                    "predicted_ite": 0.2,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                }
            ],
            "num_bins": 51,
        }
        res_too_many = self.client.post("/api/v1/uplift-curve", json=too_many_bins)
        self.assertEqual(res_too_many.status_code, 422)


if __name__ == "__main__":
    unittest.main()
