"""Unit tests for the Knee-Point Budget Detector and Diminishing Returns Analyzer."""

from dataclasses import FrozenInstanceError
import unittest
import numpy as np

from src.optimization.knee_point import (
    FrontierPoint,
    KneePointRecommendation,
    detect_frontier_knee_point,
)


class TestKneePointDetector(unittest.TestCase):
    """Test suite for knee-point detection and diminishing returns analysis."""

    def setUp(self) -> None:
        """Set up standard concave frontier data."""
        self.concave_points = [
            FrontierPoint(budget=1000.0, expected_net_profit=1000.0, expected_incremental_revenue=2000.0, targeted_customers_count=50),
            FrontierPoint(budget=2000.0, expected_net_profit=2800.0, expected_incremental_revenue=4800.0, targeted_customers_count=120),
            FrontierPoint(budget=3000.0, expected_net_profit=4200.0, expected_incremental_revenue=7200.0, targeted_customers_count=190),
            FrontierPoint(budget=4000.0, expected_net_profit=4800.0, expected_incremental_revenue=8800.0, targeted_customers_count=230),
            FrontierPoint(budget=5000.0, expected_net_profit=5000.0, expected_incremental_revenue=10000.0, targeted_customers_count=250),
        ]

    def test_knee_point_detection_concave_frontier(self) -> None:
        """Verify accurate knee point identification on standard concave profit curve."""
        rec = detect_frontier_knee_point(self.concave_points)

        self.assertIsInstance(rec, KneePointRecommendation)
        self.assertEqual(rec.recommended_budget, 3000.0)
        self.assertEqual(rec.expected_net_profit, 4200.0)
        self.assertEqual(rec.expected_incremental_revenue, 7200.0)
        self.assertEqual(rec.targeted_customers_count, 190)

        # Pre-knee slope: (4200 - 2800) / (3000 - 2000) = 1.40
        # Post-knee slope: (4800 - 4200) / (4000 - 3000) = 0.60
        self.assertAlmostEqual(rec.marginal_roi, 0.60, places=4)

        # Diminishing returns ratio: 0.60 / 1.40 ≈ 0.42857
        expected_ratio = 0.60 / 1.40
        self.assertAlmostEqual(rec.diminishing_returns_ratio, expected_ratio, places=4)

        # Explanation text validation
        self.assertIn("$3,000.00", rec.explanation)
        self.assertIn("$4,200.00", rec.explanation)
        self.assertIn("0.60", rec.explanation)
        self.assertIn("diminishing returns ratio", rec.explanation)

    def test_fewer_than_three_points_raises_value_error(self) -> None:
        """Verify ValueError is raised when fewer than 3 frontier points are provided."""
        with self.assertRaises(ValueError):
            detect_frontier_knee_point([])

        with self.assertRaises(ValueError):
            detect_frontier_knee_point([FrontierPoint(budget=100.0, expected_net_profit=50.0)])

        with self.assertRaises(ValueError):
            detect_frontier_knee_point([
                FrontierPoint(budget=100.0, expected_net_profit=50.0),
                FrontierPoint(budget=200.0, expected_net_profit=120.0),
            ])

        with self.assertRaises(ValueError):
            detect_frontier_knee_point(None)  # type: ignore[arg-type]

    def test_unsorted_frontier_points(self) -> None:
        """Verify detect_frontier_knee_point sorts unsorted input points correctly."""
        unsorted = [
            self.concave_points[3],
            self.concave_points[0],
            self.concave_points[4],
            self.concave_points[1],
            self.concave_points[2],
        ]
        rec = detect_frontier_knee_point(unsorted)
        self.assertEqual(rec.recommended_budget, 3000.0)
        self.assertEqual(rec.expected_net_profit, 4200.0)

    def test_frozen_dataclass_immutability(self) -> None:
        """Verify FrontierPoint and KneePointRecommendation instances cannot be mutated."""
        pt = FrontierPoint(budget=100.0, expected_net_profit=200.0)
        with self.assertRaises(FrozenInstanceError):
            pt.budget = 150.0  # type: ignore[misc]

        rec = detect_frontier_knee_point(self.concave_points)
        with self.assertRaises(FrozenInstanceError):
            rec.recommended_budget = 4000.0  # type: ignore[misc]

    def test_post_knee_plateau_or_negative_marginal_roi(self) -> None:
        """Verify marginal ROI calculation when profit declines after knee point."""
        declining_points = [
            FrontierPoint(budget=100.0, expected_net_profit=200.0),
            FrontierPoint(budget=200.0, expected_net_profit=500.0),
            FrontierPoint(budget=300.0, expected_net_profit=450.0),
            FrontierPoint(budget=400.0, expected_net_profit=380.0),
        ]
        rec = detect_frontier_knee_point(declining_points)
        self.assertEqual(rec.recommended_budget, 200.0)
        self.assertEqual(rec.expected_net_profit, 500.0)
        self.assertLess(rec.marginal_roi, 0.0)
        self.assertIn("zero or negative incremental profit", rec.explanation)

    def test_linear_collinear_frontier_graceful_handling(self) -> None:
        """Verify linear frontier does not crash and yields valid recommendation."""
        linear_points = [
            FrontierPoint(budget=100.0, expected_net_profit=100.0),
            FrontierPoint(budget=200.0, expected_net_profit=200.0),
            FrontierPoint(budget=300.0, expected_net_profit=300.0),
            FrontierPoint(budget=400.0, expected_net_profit=400.0),
        ]
        rec = detect_frontier_knee_point(linear_points)
        self.assertIsInstance(rec, KneePointRecommendation)
        self.assertIn(rec.recommended_budget, [200.0, 300.0])
        self.assertIsInstance(rec.explanation, str)

    def test_all_recommendation_fields_and_types(self) -> None:
        """Verify types and presence of all fields on KneePointRecommendation."""
        rec = detect_frontier_knee_point(self.concave_points)
        self.assertIsInstance(rec.recommended_budget, float)
        self.assertIsInstance(rec.expected_net_profit, float)
        self.assertIsInstance(rec.expected_incremental_revenue, float)
        self.assertIsInstance(rec.targeted_customers_count, int)
        self.assertIsInstance(rec.marginal_roi, float)
        self.assertIsInstance(rec.diminishing_returns_ratio, float)
        self.assertIsInstance(rec.explanation, str)


if __name__ == "__main__":
    unittest.main()

