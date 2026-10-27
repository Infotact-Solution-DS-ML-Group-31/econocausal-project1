"""Unit tests for Customer Segment Classifier and Executive Insight Report Generator."""

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
from src.optimization.reporting import (
    ExecutiveCampaignReport,
    SegmentSummary,
    classify_customer_segments,
    generate_executive_report,
)


class TestReporting(unittest.TestCase):
    """Test suite covering customer segment classification, report generation, and API endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_classify_customer_segments_quadrant_counts(self):
        """Verify accurate quadrant counts across Sleeping Dogs, Persuadables, Sure Things, and Lost Causes."""
        # 20 customers:
        # - 4 Sleeping Dogs: ITE < 0
        # - 6 Persuadables: ITE > 0 and (ITE * 50 - 10) > 0 (i.e. ITE > 0.20)
        # - 5 Sure Things: ITE >= 0, (ITE * 50 - 10) <= 0, and baseline_prob >= 0.50
        # - 5 Lost Causes: ITE >= 0, (ITE * 50 - 10) <= 0, and baseline_prob < 0.50
        baseline_probs = np.array(
            [0.20, 0.30, 0.40, 0.10]  # Sleeping Dogs
            + [0.15, 0.25, 0.35, 0.45, 0.10, 0.20]  # Persuadables
            + [0.55, 0.60, 0.70, 0.80, 0.90]  # Sure Things
            + [0.10, 0.20, 0.30, 0.40, 0.45]  # Lost Causes
        )
        treatment_effects = np.array(
            [-0.10, -0.05, -0.15, -0.02]  # Sleeping Dogs
            + [0.30, 0.40, 0.25, 0.35, 0.50, 0.22]  # Persuadables (net_gain > 0)
            + [0.05, 0.10, 0.08, 0.02, 0.15]  # Sure Things (net_gain <= 0, prob >= 0.5)
            + [0.05, 0.10, 0.02, 0.12, 0.00]  # Lost Causes (net_gain <= 0, prob < 0.5)
        )
        treatment_costs = np.full(20, 10.0)
        customer_values = np.full(20, 50.0)

        segments = classify_customer_segments(
            baseline_probs=baseline_probs,
            treatment_effects=treatment_effects,
            treatment_costs=treatment_costs,
            customer_values=customer_values,
        )

        self.assertIsInstance(segments, list)
        self.assertEqual(len(segments), 4)

        seg_dict = {s.segment_name: s for s in segments}
        self.assertIn("Persuadables", seg_dict)
        self.assertIn("Sure Things", seg_dict)
        self.assertIn("Lost Causes", seg_dict)
        self.assertIn("Sleeping Dogs", seg_dict)

        # Check customer counts
        self.assertEqual(seg_dict["Persuadables"].customer_count, 6)
        self.assertEqual(seg_dict["Sure Things"].customer_count, 5)
        self.assertEqual(seg_dict["Lost Causes"].customer_count, 5)
        self.assertEqual(seg_dict["Sleeping Dogs"].customer_count, 4)

        # Assert sum(s.customer_count) == N
        self.assertEqual(sum(s.customer_count for s in segments), 20)

        # Verify population percentages
        self.assertAlmostEqual(seg_dict["Persuadables"].population_pct, 30.0, places=2)
        self.assertAlmostEqual(seg_dict["Sure Things"].population_pct, 25.0, places=2)
        self.assertAlmostEqual(seg_dict["Lost Causes"].population_pct, 25.0, places=2)
        self.assertAlmostEqual(seg_dict["Sleeping Dogs"].population_pct, 20.0, places=2)

        # Verify mean ITEs and baseline probs
        self.assertGreater(seg_dict["Persuadables"].mean_ite, 0.20)
        self.assertLess(seg_dict["Sleeping Dogs"].mean_ite, 0.0)
        self.assertGreaterEqual(seg_dict["Sure Things"].mean_baseline_prob, 0.50)
        self.assertLess(seg_dict["Lost Causes"].mean_baseline_prob, 0.50)

        # Verify recommended actions
        self.assertIn("Primary target", seg_dict["Persuadables"].recommended_action)
        self.assertIn("discount unnecessary", seg_dict["Sure Things"].recommended_action)
        self.assertIn("Holdout", seg_dict["Lost Causes"].recommended_action)
        self.assertIn("Do not discount", seg_dict["Sleeping Dogs"].recommended_action)

    def test_classify_customer_segments_validation(self):
        """Verify robust input validation on malformed inputs, NaNs, and bounds violations."""
        # Length mismatch
        with self.assertRaises(ValueError):
            classify_customer_segments(
                baseline_probs=np.array([0.2, 0.3]),
                treatment_effects=np.array([0.1]),
                treatment_costs=np.array([10.0, 10.0]),
                customer_values=np.array([50.0, 50.0]),
            )

        # 2D input
        with self.assertRaises(ValueError):
            classify_customer_segments(
                baseline_probs=np.array([[0.2, 0.3]]),
                treatment_effects=np.array([[0.1, 0.2]]),
                treatment_costs=np.array([[10.0, 10.0]]),
                customer_values=np.array([[50.0, 50.0]]),
            )

        # NaN values
        with self.assertRaises(ValueError):
            classify_customer_segments(
                baseline_probs=np.array([0.2, np.nan]),
                treatment_effects=np.array([0.1, 0.2]),
                treatment_costs=np.array([10.0, 10.0]),
                customer_values=np.array([50.0, 50.0]),
            )

        # Probability out of bounds
        with self.assertRaises(ValueError):
            classify_customer_segments(
                baseline_probs=np.array([1.2, 0.3]),
                treatment_effects=np.array([0.1, 0.2]),
                treatment_costs=np.array([10.0, 10.0]),
                customer_values=np.array([50.0, 50.0]),
            )

        # Empty cohort
        empty_segments = classify_customer_segments(
            baseline_probs=np.array([]),
            treatment_effects=np.array([]),
            treatment_costs=np.array([]),
            customer_values=np.array([]),
        )
        self.assertEqual(len(empty_segments), 4)
        self.assertEqual(sum(s.customer_count for s in empty_segments), 0)

    def test_frozen_dataclass_immutability(self):
        """Verify SegmentSummary and ExecutiveCampaignReport are immutable frozen dataclasses."""
        seg = SegmentSummary(
            segment_name="Persuadables",
            customer_count=10,
            population_pct=50.0,
            mean_ite=0.15,
            mean_baseline_prob=0.25,
            recommended_action="Test action",
        )
        with self.assertRaises(FrozenInstanceError):
            seg.customer_count = 20  # type: ignore

        report = ExecutiveCampaignReport(
            total_customers=10,
            budget=100.0,
            total_spend=50.0,
            expected_incremental_revenue=150.0,
            expected_net_profit=100.0,
            roi_pct=200.0,
            savings_vs_blanket=25.0,
            segments=[seg],
            headline_summary="Summary headline",
            bullet_insights=["Insight 1"],
        )
        with self.assertRaises(FrozenInstanceError):
            report.total_customers = 20  # type: ignore

    def test_generate_executive_report(self):
        """Verify generate_executive_report runs SciPy and Blanket solvers, computing valid metrics."""
        mock_customers = generate_mock_customers(num_customers=100, seed=42)
        request = OptimizationRequest(budget=250.0, customers=mock_customers)

        report = generate_executive_report(request)

        self.assertIsInstance(report, ExecutiveCampaignReport)
        self.assertEqual(report.total_customers, 100)
        self.assertEqual(report.budget, 250.0)
        self.assertLessEqual(report.total_spend, 250.0)
        self.assertGreaterEqual(report.savings_vs_blanket, 0.0)
        self.assertGreater(len(report.headline_summary), 0)
        self.assertIn("Causal optimization generates", report.headline_summary)
        self.assertGreaterEqual(len(report.bullet_insights), 3)

        # Quadrant segments
        self.assertEqual(len(report.segments), 4)
        self.assertEqual(sum(s.customer_count for s in report.segments), 100)

    def test_generate_executive_report_sleeping_dogs_cohort(self):
        """Verify report generation on all-negative ITE cohort allocates zero budget."""
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
        report = generate_executive_report(request)

        self.assertEqual(report.total_customers, 10)
        self.assertEqual(report.total_spend, 0.0)
        self.assertEqual(report.expected_net_profit, 0.0)
        self.assertGreaterEqual(report.savings_vs_blanket, 0.0)

        seg_dict = {s.segment_name: s for s in report.segments}
        self.assertEqual(seg_dict["Sleeping Dogs"].customer_count, 10)

    def test_post_executive_report_endpoint_success(self):
        """Test POST /api/v1/executive-report via TestClient asserting HTTP 200 and savings_vs_blanket >= 0."""
        mock_customers = generate_mock_customers(num_customers=80, seed=123)
        payload = {
            "budget": 200.0,
            "objective": "maximize_profit",
            "min_spend_per_user": 0.0,
            "customers": [c.model_dump() for c in mock_customers],
        }

        response = self.client.post("/api/v1/executive-report", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["total_customers"], 80)
        self.assertEqual(data["budget"], 200.0)
        self.assertLessEqual(data["total_spend"], 200.0)
        self.assertGreaterEqual(data["savings_vs_blanket"], 0.0)
        self.assertIn("Causal optimization generates", data["headline_summary"])
        self.assertGreaterEqual(len(data["bullet_insights"]), 3)
        self.assertEqual(len(data["segments"]), 4)
        self.assertEqual(sum(s["customer_count"] for s in data["segments"]), 80)

    def test_post_executive_report_validation_errors(self):
        """Verify API schema validation rejects duplicate customer IDs and invalid budgets."""
        # Duplicate customer_id
        dup_payload = {
            "budget": 100.0,
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
        res = self.client.post("/api/v1/executive-report", json=dup_payload)
        self.assertEqual(res.status_code, 422)

        # Empty customers list
        empty_payload = {"budget": 100.0, "customers": []}
        res = self.client.post("/api/v1/executive-report", json=empty_payload)
        self.assertEqual(res.status_code, 422)

        # Negative budget
        neg_budget = {
            "budget": -50.0,
            "customers": [
                {
                    "customer_id": "cust_1",
                    "baseline_conversion_prob": 0.2,
                    "predicted_ite": 0.1,
                    "treatment_cost": 10.0,
                    "customer_value": 50.0,
                }
            ],
        }
        res = self.client.post("/api/v1/executive-report", json=neg_budget)
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()

