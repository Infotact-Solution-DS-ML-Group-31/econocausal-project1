import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.schemas import OptimizationObjective
from src.optimization.baseline import BlanketDiscountSolver
from src.optimization.solver import SciPyConstrainedSolver


class TestSciPyConstrainedSolver(unittest.TestCase):
    """Test suite covering LP optimization, business objectives, and edge cases."""

    def setUp(self):
        self.te = np.array([-0.10, 0.02, 0.35, 0.28, -0.05])
        self.tc = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.cv = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    def test_maximize_profit_objective(self):
        """Verify MAXIMIZE_PROFIT targets only profitable persuadables within budget."""
        solver = SciPyConstrainedSolver(
            budget=20.0,
            objective=OptimizationObjective.MAXIMIZE_PROFIT,
            integer_rounding=True,
        )
        res = solver.solve(self.te, self.tc, customer_values=self.cv)

        self.assertTrue(res.success)
        self.assertEqual(res.metadata["solver"], "SciPyConstrainedSolver (HiGHS)")
        self.assertEqual(res.metadata["objective"], "maximize_profit")
        self.assertEqual(res.metadata["targeted_count"], 2)
        self.assertEqual(res.metadata["expected_net_profit"], 43.0)
        self.assertEqual(res.total_spend, 20.0)
        self.assertEqual(res.expected_incremental_revenue, 63.0)
        # Should allocate to index 2 and index 3
        np.testing.assert_array_equal(res.allocations, np.array([0, 0, 1, 1, 0]))

    def test_maximize_revenue_objective(self):
        """Verify MAXIMIZE_REVENUE prioritizes highest incremental revenue (te * cv)."""
        te = np.array([0.10, 0.30, -0.05])
        tc = np.array([10.0, 10.0, 10.0])
        cv = np.array([500.0, 100.0, 200.0])  # Rev deltas: 50.0, 30.0, -10.0

        solver = SciPyConstrainedSolver(
            budget=10.0,
            objective=OptimizationObjective.MAXIMIZE_REVENUE,
            integer_rounding=True,
        )
        res = solver.solve(te, tc, customer_values=cv)

        self.assertTrue(res.success)
        self.assertEqual(res.metadata["objective"], "maximize_revenue")
        self.assertEqual(res.metadata["targeted_count"], 1)
        # Should prioritize customer 0 because revenue delta ($50) > customer 1 ($30)
        np.testing.assert_array_equal(res.allocations, np.array([1, 0, 0]))
        self.assertEqual(res.total_spend, 10.0)

    def test_maximize_conversions_objective(self):
        """Verify MAXIMIZE_CONVERSIONS prioritizes highest uplift (ITE) regardless of AOV."""
        te = np.array([0.10, 0.40, 0.05])
        tc = np.array([10.0, 10.0, 10.0])
        cv = np.array([1000.0, 50.0, 200.0])

        solver = SciPyConstrainedSolver(
            budget=10.0,
            objective=OptimizationObjective.MAXIMIZE_CONVERSIONS,
            integer_rounding=True,
        )
        res = solver.solve(te, tc, customer_values=cv)

        self.assertTrue(res.success)
        self.assertEqual(res.metadata["objective"], "maximize_conversions")
        self.assertEqual(res.metadata["targeted_count"], 1)
        # Should prioritize customer 1 because ITE 0.40 > 0.10
        np.testing.assert_array_equal(res.allocations, np.array([0, 1, 0]))

    def test_edge_case_zero_budget(self):
        """Budget = 0.0 must immediately return 0 allocations and 0 spend."""
        solver = SciPyConstrainedSolver(budget=0.0)
        res = solver.solve(self.te, self.tc, customer_values=self.cv)

        self.assertTrue(res.success)
        self.assertEqual(res.total_spend, 0.0)
        self.assertEqual(res.expected_incremental_revenue, 0.0)
        self.assertEqual(res.metadata["targeted_count"], 0)
        self.assertEqual(res.metadata["expected_net_profit"], 0.0)
        np.testing.assert_array_equal(res.allocations, np.zeros(5, dtype=np.int32))

    def test_edge_case_budget_below_min_treatment_cost(self):
        """Budget < min(cost) must return status 'Budget below minimum treatment cost; zero allocations made'."""
        costs = np.array([15.0, 20.0, 25.0])
        te = np.array([0.2, 0.3, 0.4])
        cv = np.array([100.0, 100.0, 100.0])

        solver = SciPyConstrainedSolver(budget=10.0)
        res = solver.solve(te, costs, customer_values=cv)

        self.assertTrue(res.success)
        self.assertIn("Budget below minimum treatment cost; zero allocations made", res.status_message)
        self.assertEqual(res.total_spend, 0.0)
        self.assertEqual(res.metadata["targeted_count"], 0)
        np.testing.assert_array_equal(res.allocations, np.zeros(3, dtype=np.int32))

    def test_edge_case_unconstrained_budget_ceiling(self):
        """Large budget (100,000.0) should target all eligible candidates without targeting sleeping dogs."""
        solver = SciPyConstrainedSolver(budget=100000.0)
        res = solver.solve(self.te, self.tc, customer_values=self.cv)

        self.assertTrue(res.success)
        # In self.te, only indices 2 and 3 have positive net profit (0.35*100-10=25, 0.28*100-10=18)
        # Index 1 has te=0.02 -> net profit = 0.02*100-10 = -8.0 (unprofitable)
        # Indices 0 and 4 are sleeping dogs (negative ITE)
        self.assertEqual(res.metadata["targeted_count"], 2)
        np.testing.assert_array_equal(res.allocations, np.array([0, 0, 1, 1, 0]))
        self.assertEqual(res.total_spend, 20.0)
        self.assertLessEqual(res.total_spend, 100000.0)

    def test_edge_case_all_negative_ite_cohort(self):
        """All-negative ITE cohort (all sleeping dogs) must result in zero allocations."""
        all_neg_te = np.array([-0.05, -0.12, -0.25, -0.01])
        tc = np.array([10.0, 10.0, 10.0, 10.0])
        cv = np.array([100.0, 100.0, 100.0, 100.0])

        solver = SciPyConstrainedSolver(budget=50.0)
        res = solver.solve(all_neg_te, tc, customer_values=cv)

        self.assertTrue(res.success)
        self.assertEqual(res.total_spend, 0.0)
        self.assertEqual(res.metadata["targeted_count"], 0)
        np.testing.assert_array_equal(res.allocations, np.zeros(4, dtype=np.int32))

    def test_scipy_solver_matches_or_beats_blanket_discount(self):
        """SciPyConstrainedSolver must match or outperform BlanketDiscountSolver in expected net profit."""
        budget = 20.0
        blanket_res = BlanketDiscountSolver(budget=budget).solve(
            self.te, self.tc, customer_values=self.cv
        )
        scipy_res = SciPyConstrainedSolver(budget=budget).solve(
            self.te, self.tc, customer_values=self.cv
        )

        blanket_profit = blanket_res.metadata["expected_net_profit"]
        scipy_profit = scipy_res.metadata["expected_net_profit"]

        self.assertGreater(scipy_profit, blanket_profit)
        self.assertEqual(blanket_profit, -28.0)
        self.assertEqual(scipy_profit, 43.0)
        self.assertLessEqual(scipy_res.total_spend, budget)

    def test_continuous_vs_integer_allocations(self):
        """Verify non-integer solver produces continuous LP solutions within budget."""
        te = np.array([0.30, 0.20])
        tc = np.array([10.0, 10.0])
        cv = np.array([100.0, 100.0])
        # Budget = 15 allows 1 full and 1 half allocation in LP relaxation
        solver = SciPyConstrainedSolver(budget=15.0, integer_rounding=False)
        res = solver.solve(te, tc, customer_values=cv)

        self.assertTrue(res.success)
        self.assertAlmostEqual(res.total_spend, 15.0, places=4)
        np.testing.assert_allclose(res.allocations, np.array([1.0, 0.5]), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
