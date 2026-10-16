"""Unit tests for covariate and treatment effect drift monitoring."""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.optimization.drift import (
    DriftReport,
    calculate_psi,
    evaluate_distribution_drift,
)


class TestDriftMonitoring(unittest.TestCase):
    """Test suite for calculate_psi and evaluate_distribution_drift."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_identical_distributions_no_drift(self):
        """Identical distributions should have zero PSI and severity == 'none'."""
        ref = self.rng.normal(0.10, 0.02, 500)
        cur = ref.copy()

        psi = calculate_psi(ref, cur)
        self.assertEqual(psi, 0.0)

        report = evaluate_distribution_drift(ref, cur)
        self.assertIsInstance(report, DriftReport)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.severity, "none")
        self.assertEqual(report.psi_score, 0.0)
        self.assertEqual(report.ks_statistic, 0.0)
        self.assertEqual(report.ks_pvalue, 1.0)
        self.assertIn("No significant distribution drift detected", report.recommendation)

    def test_similar_distributions_no_drift(self):
        """Samples from identical generating process should have low PSI and no drift."""
        ref = self.rng.normal(0.15, 0.03, 500)
        cur = self.rng.normal(0.15, 0.03, 500)

        psi = calculate_psi(ref, cur)
        self.assertLess(psi, 0.10)

        report = evaluate_distribution_drift(ref, cur)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.severity, "none")
        self.assertLess(report.psi_score, 0.10)
        self.assertIn("No significant distribution drift detected", report.recommendation)

    def test_shifted_distributions_critical_drift(self):
        """Severely shifted distributions must trigger critical severity and drift_detected == True."""
        ref = self.rng.normal(0.05, 0.02, 500)
        cur = self.rng.normal(0.25, 0.05, 500)

        psi = calculate_psi(ref, cur)
        self.assertGreaterEqual(psi, 0.25)

        report = evaluate_distribution_drift(ref, cur)
        self.assertTrue(report.drift_detected)
        self.assertEqual(report.severity, "critical")
        self.assertGreaterEqual(report.psi_score, 0.25)
        self.assertLess(report.ks_pvalue, 0.05)
        self.assertIn("Critical distribution shift detected", report.recommendation)
        self.assertIn("retrain EconML Double ML model", report.recommendation)

    def test_moderate_drift_classification(self):
        """Moderate shift with psi between 0.10 and 0.25 without KS trigger should be 'moderate'."""
        ref = self.rng.normal(0.10, 0.02, 1000)
        cur = self.rng.normal(0.108, 0.02, 1000)

        # Use very strict ks_alpha to isolate PSI-based moderate severity
        report = evaluate_distribution_drift(ref, cur, psi_threshold=0.25, ks_alpha=1e-15)
        self.assertTrue(report.drift_detected)
        self.assertEqual(report.severity, "moderate")
        self.assertGreaterEqual(report.psi_score, 0.10)
        self.assertLess(report.psi_score, 0.25)
        self.assertIn("Moderate distribution drift detected", report.recommendation)

    def test_constant_distributions_return_zero_psi(self):
        """Distributions with zero variance (min == max) return 0.0 PSI."""
        ref = np.full(50, 0.15)
        cur = np.full(50, 0.15)

        psi = calculate_psi(ref, cur)
        self.assertEqual(psi, 0.0)

        report = evaluate_distribution_drift(ref, cur)
        self.assertEqual(report.severity, "none")
        self.assertFalse(report.drift_detected)

    def test_empty_array_rejection(self):
        """Empty arrays for reference or current must raise ValueError."""
        valid = np.array([0.1, 0.2, 0.3])
        empty = np.array([])

        with self.assertRaises(ValueError) as ctx1:
            calculate_psi(empty, valid)
        self.assertIn("empty", str(ctx1.exception).lower())

        with self.assertRaises(ValueError) as ctx2:
            calculate_psi(valid, empty)
        self.assertIn("empty", str(ctx2.exception).lower())

        with self.assertRaises(ValueError) as ctx3:
            evaluate_distribution_drift(empty, valid)
        self.assertIn("empty", str(ctx3.exception).lower())

    def test_nan_and_inf_rejection(self):
        """Arrays containing NaN or Inf must raise ValueError."""
        valid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        nan_arr = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
        inf_arr = np.array([0.1, np.inf, 0.3, 0.4, 0.5])

        with self.assertRaises(ValueError) as ctx1:
            calculate_psi(nan_arr, valid)
        self.assertIn("nan", str(ctx1.exception).lower())

        with self.assertRaises(ValueError) as ctx2:
            calculate_psi(valid, nan_arr)
        self.assertIn("nan", str(ctx2.exception).lower())

        with self.assertRaises(ValueError) as ctx3:
            calculate_psi(inf_arr, valid)
        self.assertIn("inf", str(ctx3.exception).lower())

        with self.assertRaises(ValueError) as ctx4:
            evaluate_distribution_drift(valid, nan_arr)
        self.assertIn("nan", str(ctx4.exception).lower())

    def test_invalid_dimensions_and_bins(self):
        """2D arrays and non-positive bins must raise ValueError."""
        valid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        two_d = np.array([[0.1, 0.2], [0.3, 0.4]])

        with self.assertRaises(ValueError):
            calculate_psi(two_d, valid)

        with self.assertRaises(ValueError):
            calculate_psi(valid, valid, bins=0)

        with self.assertRaises(ValueError):
            calculate_psi(valid, valid, bins=-5)


if __name__ == "__main__":
    unittest.main()
