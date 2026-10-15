"""Unit tests for constraints and baseline heuristic solvers."""

import unittest
import numpy as np
from src.optimization.baseline import BlanketDiscountSolver, GreedyROISolver
from src.optimization.constraints import prepare_feasible_inputs, validate_solver_inputs


class TestOptimizationBaselineAndConstraints(unittest.TestCase):
    def test_2d_multi_treatment_validation(self):
        te = np.array([[0.05, 0.12], [0.15, 0.28], [-0.02, -0.05]])
        tc = np.array([10.0, 20.0])
        res_te, res_tc = validate_solver_inputs(te, tc, budget=100.0)
        self.assertEqual(res_te.shape, (3, 2))
        self.assertEqual(res_tc.shape, (2,))

    def test_prepare_feasible_inputs_filters_sleeping_dogs(self):
        te = np.array([0.30, 0.05, -0.10])
        tc = np.array([10.0, 10.0, 10.0])
        cv = np.array([100.0, 100.0, 100.0])
        feasible = prepare_feasible_inputs(te, tc, cv, budget=50.0)
        np.testing.assert_array_equal(feasible.eligible_mask, np.array([True, False, False]))

    def test_greedy_roi_outperforms_blanket_discount(self):
        # 5 customers: 2 sleeping dogs, 1 low uplift, 2 strong persuadables
        te = np.array([-0.10, 0.02, 0.35, 0.28, -0.05])
        tc = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        cv = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        blanket_res = BlanketDiscountSolver(budget=20.0).solve(te, tc, customer_values=cv)
        greedy_res = GreedyROISolver(budget=20.0).solve(te, tc, customer_values=cv)

        # Blanket wastes $20 on first two customers (net profit = -28.0)
        self.assertEqual(blanket_res.metadata["expected_net_profit"], -28.0)
        # Greedy targets only indices 2 and 3 (net profit = +43.0)
        self.assertEqual(greedy_res.metadata["expected_net_profit"], 43.0)
        np.testing.assert_array_equal(greedy_res.allocations, np.array([0, 0, 1, 1, 0]))
        self.assertGreater(
            greedy_res.metadata["expected_net_profit"],
            blanket_res.metadata["expected_net_profit"],
        )

    def test_zero_budget_allocates_nothing(self):
        te = np.array([0.30, 0.25])
        tc = np.array([10.0, 10.0])
        res = GreedyROISolver(budget=0.0).solve(te, tc)
        self.assertEqual(res.total_spend, 0.0)
        self.assertEqual(res.metadata["targeted_count"], 0)


if __name__ == "__main__":
    unittest.main()