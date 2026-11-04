"""Numerical edge-case fuzzing and boundary stress test suite for optimization algorithms."""

import unittest
import numpy as np

from src.api.schemas import OptimizationObjective
from src.optimization.knee_point import (
    FrontierPoint,
    KneePointRecommendation,
    detect_frontier_knee_point,
)
from src.optimization.solver import SciPyConstrainedSolver


class TestNumericalFuzzAndBoundaryBounds(unittest.TestCase):
    """Stress testing optimization solvers and knee-point detectors under extreme numerical regimes."""

    def test_fuzz_randomized_cohorts_budget_compliance(self) -> None:
        """Fuzz 30 randomized cohorts across objectives, asserting budget compliance and binary allocation."""
        rng = np.random.default_rng(seed=42)
        objectives = [
            OptimizationObjective.MAXIMIZE_PROFIT,
            OptimizationObjective.MAXIMIZE_REVENUE,
            OptimizationObjective.MAXIMIZE_CONVERSIONS,
        ]

        for trial in range(30):
            n_customers = int(rng.integers(10, 201))
            budget = float(rng.uniform(5.0, 2000.0))
            ite = rng.uniform(-0.5, 0.8, size=n_customers)
            costs = rng.uniform(1.0, 50.0, size=n_customers)
            values = rng.uniform(10.0, 500.0, size=n_customers)

            for obj in objectives:
                solver = SciPyConstrainedSolver(
                    budget=budget,
                    objective=obj,
                    integer_rounding=True,
                )
                result = solver.solve(
                    treatment_effects=ite,
                    treatment_costs=costs,
                    customer_values=values,
                )

                self.assertTrue(result.success, f"Trial {trial} ({obj.value}) failed solver execution")
                self.assertLessEqual(
                    result.total_spend,
                    budget + 1e-6,
                    f"Trial {trial} ({obj.value}) exceeded budget {budget}: spend={result.total_spend}",
                )
                self.assertTrue(
                    np.all(np.isin(result.allocations, [0, 1])),
                    f"Trial {trial} ({obj.value}) produced non-binary allocations",
                )
                expected_targeted = int(np.sum(result.allocations))
                self.assertEqual(
                    result.metadata["targeted_count"],
                    expected_targeted,
                    f"Trial {trial} ({obj.value}) targeted_count mismatch",
                )

    def test_extreme_budget_scaling(self) -> None:
        """Verify solver robustness under micro-budgets, threshold budgets, and massive budgets."""
        n_customers = 50
        ite = np.full(n_customers, 0.25)
        costs = np.linspace(10.0, 50.0, n_customers)
        # customer_value = 500.0 ensures all positive ITE customers are profitable (0.25 * 500 = 125 > 50)
        values = np.full(n_customers, 500.0)

        # 1. Micro-budgets well below minimum treatment cost ($10.0)
        for micro_budget in [1e-6, 0.01, 0.5]:
            solver = SciPyConstrainedSolver(budget=micro_budget, integer_rounding=True)
            res = solver.solve(ite, costs, customer_values=values)
            self.assertTrue(res.success)
            self.assertEqual(res.total_spend, 0.0)
            self.assertEqual(res.metadata["targeted_count"], 0)
            self.assertEqual(int(np.sum(res.allocations)), 0)

        # 2. Exact single-customer threshold budget (min cost = 10.0)
        threshold_budget = float(np.min(costs))
        solver_thresh = SciPyConstrainedSolver(budget=threshold_budget, integer_rounding=True)
        res_thresh = solver_thresh.solve(ite, costs, customer_values=values)
        self.assertTrue(res_thresh.success)
        self.assertLessEqual(res_thresh.total_spend, threshold_budget + 1e-6)
        self.assertEqual(res_thresh.metadata["targeted_count"], 1)
        self.assertEqual(int(np.sum(res_thresh.allocations)), 1)
        self.assertEqual(res_thresh.total_spend, threshold_budget)

        # 3. Massive budget (B = 10^7) exceeding total campaign cost
        massive_budget = 1e7
        # Default safety limit (1,000,000.0) raises ValueError
        with self.assertRaises(ValueError):
            SciPyConstrainedSolver(budget=massive_budget, integer_rounding=True).solve(ite, costs, customer_values=values)

        # When authorized with expanded budget limit, solves without floating-point overflow or crashes
        solver_massive = SciPyConstrainedSolver(budget=massive_budget, max_budget_limit=1e8, integer_rounding=True)
        res_massive = solver_massive.solve(ite, costs, customer_values=values)
        self.assertTrue(res_massive.success)
        self.assertLessEqual(res_massive.total_spend, massive_budget)
        total_cohort_cost = float(np.sum(costs))
        self.assertAlmostEqual(res_massive.total_spend, total_cohort_cost, places=3)
        self.assertEqual(res_massive.metadata["targeted_count"], n_customers)

    def test_degenerate_customer_cohorts(self) -> None:
        """Verify handling of all-negative, all-zero, and identical customer cohorts."""
        n_customers = 30
        costs = np.full(n_customers, 15.0)
        values = np.full(n_customers, 80.0)

        # 1. All-negative ITE cohort (100% Sleeping Dogs)
        neg_ite = np.linspace(-0.5, -0.01, n_customers)
        solver_neg = SciPyConstrainedSolver(budget=300.0, integer_rounding=True)
        res_neg = solver_neg.solve(neg_ite, costs, customer_values=values)
        self.assertTrue(res_neg.success)
        self.assertEqual(res_neg.total_spend, 0.0)
        self.assertEqual(res_neg.metadata["targeted_count"], 0)
        self.assertEqual(int(np.sum(res_neg.allocations)), 0)

        # 2. All-zero ITE cohort (Zero Uplift)
        zero_ite = np.zeros(n_customers)
        solver_zero = SciPyConstrainedSolver(budget=300.0, integer_rounding=True)
        res_zero = solver_zero.solve(zero_ite, costs, customer_values=values)
        self.assertTrue(res_zero.success)
        self.assertEqual(res_zero.total_spend, 0.0)
        self.assertEqual(res_zero.metadata["targeted_count"], 0)
        self.assertEqual(int(np.sum(res_zero.allocations)), 0)

        # 3. Identical customer cohort (tie-breaking behavior)
        ident_ite = np.full(n_customers, 0.30)  # Net profit: 0.30 * 80 - 15 = 9 > 0
        budget = 75.0  # Budget covers exactly 5 customers (5 * 15 = 75)
        solver_ident = SciPyConstrainedSolver(budget=budget, integer_rounding=True)
        res_ident = solver_ident.solve(ident_ite, costs, customer_values=values)
        self.assertTrue(res_ident.success)
        self.assertAlmostEqual(res_ident.total_spend, 75.0, places=4)
        self.assertEqual(res_ident.metadata["targeted_count"], 5)
        self.assertEqual(int(np.sum(res_ident.allocations)), 5)
        self.assertTrue(np.all(np.isin(res_ident.allocations, [0, 1])))

    def test_2d_multi_tier_fuzz_mutual_exclusivity(self) -> None:
        """Generate 20 randomized 2D cohorts and verify mutual exclusivity and budget compliance."""
        rng = np.random.default_rng(seed=123)
        tier_costs = np.array([5.0, 15.0, 30.0])

        for trial in range(20):
            n_customers = int(rng.integers(10, 101))
            budget = float(rng.uniform(20.0, 1000.0))
            te_2d = rng.uniform(-0.2, 0.6, size=(n_customers, 3))
            values = rng.uniform(20.0, 200.0, size=n_customers)

            solver = SciPyConstrainedSolver(
                budget=budget,
                objective=OptimizationObjective.MAXIMIZE_PROFIT,
                integer_rounding=True,
            )
            result = solver.solve(
                treatment_effects=te_2d,
                treatment_costs=tier_costs,
                customer_values=values,
            )

            self.assertTrue(result.success, f"Multi-tier trial {trial} failed")
            self.assertEqual(result.allocations.shape, (n_customers, 3))
            # Mutual exclusivity constraint: at most one treatment tier per customer
            customer_assignment_sums = result.allocations.sum(axis=1)
            self.assertTrue(
                np.all(customer_assignment_sums <= 1),
                f"Multi-tier trial {trial} violated mutual exclusivity: max assigned={np.max(customer_assignment_sums)}",
            )
            self.assertTrue(
                np.all(np.isin(result.allocations, [0, 1])),
                f"Multi-tier trial {trial} produced non-binary allocations",
            )
            self.assertLessEqual(
                result.total_spend,
                budget + 1e-6,
                f"Multi-tier trial {trial} exceeded budget: spend={result.total_spend}, budget={budget}",
            )

    def test_knee_point_detector_fuzz_collinear_and_flat(self) -> None:
        """Test knee point detector on flat, perfectly linear, and noisy non-monotonic frontiers."""
        # 1. Completely flat profit frontier (zero variation in profit)
        flat_points = [
            FrontierPoint(budget=b, expected_net_profit=500.0, expected_incremental_revenue=1000.0, targeted_customers_count=20)
            for b in [100.0, 200.0, 300.0, 400.0, 500.0]
        ]
        rec_flat = detect_frontier_knee_point(flat_points)
        self.assertIsInstance(rec_flat, KneePointRecommendation)
        self.assertIn(rec_flat.recommended_budget, [100.0, 200.0, 300.0, 400.0, 500.0])
        self.assertIsInstance(rec_flat.explanation, str)
        self.assertGreater(len(rec_flat.explanation), 0)

        # 2. Perfectly linear profit frontier (constant slope)
        linear_points = [
            FrontierPoint(budget=b, expected_net_profit=2.0 * b, expected_incremental_revenue=3.0 * b, targeted_customers_count=int(b // 10))
            for b in [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
        ]
        rec_linear = detect_frontier_knee_point(linear_points)
        self.assertIsInstance(rec_linear, KneePointRecommendation)
        self.assertIn(rec_linear.recommended_budget, [p.budget for p in linear_points])
        self.assertIsInstance(rec_linear.explanation, str)

        # 3. Noisy non-monotonic frontiers (20 randomized iterations)
        rng = np.random.default_rng(seed=999)
        for noise_trial in range(20):
            n_pts = int(rng.integers(4, 25))
            budgets = np.sort(rng.uniform(50.0, 1000.0, size=n_pts))
            # Concave backbone with random perturbations
            base_curve = 800.0 * (1.0 - np.exp(-budgets / 250.0))
            noisy_profit = base_curve + rng.normal(0.0, 30.0, size=n_pts)

            frontier = [
                FrontierPoint(
                    budget=float(b),
                    expected_net_profit=float(p),
                    expected_incremental_revenue=float(p + b),
                    targeted_customers_count=int(b // 10),
                )
                for b, p in zip(budgets, noisy_profit)
            ]
            rec_noisy = detect_frontier_knee_point(frontier)
            self.assertIsInstance(rec_noisy, KneePointRecommendation)
            self.assertGreaterEqual(rec_noisy.recommended_budget, float(budgets[0]))
            self.assertLessEqual(rec_noisy.recommended_budget, float(budgets[-1]))
            self.assertIsInstance(rec_noisy.explanation, str)
            self.assertGreater(len(rec_noisy.explanation), 0)


if __name__ == "__main__":
    unittest.main()
