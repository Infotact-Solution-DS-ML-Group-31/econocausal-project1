"""Baseline heuristic discount allocation strategies for benchmarking."""

from typing import Optional
import numpy as np
from src.optimization.base import BaseAllocationSolver, OptimizationResult
from src.optimization.constraints import prepare_feasible_inputs, validate_solver_inputs


class BlanketDiscountSolver(BaseAllocationSolver):
    """Naive baseline: allocates discounts sequentially until budget is exhausted."""

    def solve(
        self,
        treatment_effects: np.ndarray,
        treatment_costs: np.ndarray,
        customer_ids: Optional[np.ndarray] = None,
        customer_values: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        te, tc = validate_solver_inputs(
            treatment_effects=treatment_effects,
            treatment_costs=treatment_costs,
            budget=self.budget,
            customer_values=customer_values,
        )
        if te.ndim != 1:
            raise ValueError("BlanketDiscountSolver currently supports 1D treatment vectors.")

        cv = (
            np.asarray(customer_values, dtype=np.float64)
            if customer_values is not None
            else np.ones_like(te) * 50.0
        )

        allocations = np.zeros(te.shape[0], dtype=np.int32)
        total_spend = 0.0

        for i in range(te.shape[0]):
            cost = float(tc[i])
            if total_spend + cost <= self.budget:
                allocations[i] = 1
                total_spend += cost

        inc_rev = float(np.sum((te * cv) * allocations))
        net_profit = inc_rev - total_spend

        return OptimizationResult(
            success=True,
            status_message="Blanket allocation completed",
            total_spend=round(total_spend, 4),
            expected_incremental_revenue=round(inc_rev, 4),
            allocations=allocations,
            metadata={
                "solver": "BlanketDiscountSolver",
                "targeted_count": int(np.sum(allocations)),
                "expected_net_profit": round(net_profit, 4),
            },
        )


class GreedyROISolver(BaseAllocationSolver):
    """Greedy benchmark: ranks profitable Persuadables by net ROI per dollar spent."""

    def solve(
        self,
        treatment_effects: np.ndarray,
        treatment_costs: np.ndarray,
        customer_ids: Optional[np.ndarray] = None,
        customer_values: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        cv = (
            np.asarray(customer_values, dtype=np.float64)
            if customer_values is not None
            else np.ones_like(np.asarray(treatment_effects, dtype=np.float64)) * 50.0
        )

        feasible = prepare_feasible_inputs(
            treatment_effects=treatment_effects,
            treatment_costs=treatment_costs,
            customer_values=cv,
            budget=self.budget,
            require_positive_net_profit=True,
        )

        te = feasible.treatment_effects
        tc = feasible.treatment_costs
        if te.ndim != 1:
            raise ValueError("GreedyROISolver currently supports 1D treatment vectors.")

        allocations = np.zeros(te.shape[0], dtype=np.int32)
        eligible_indices = np.where(feasible.eligible_mask)[0]

        if eligible_indices.size == 0 or self.budget == 0.0:
            return OptimizationResult(
                success=True,
                status_message="No eligible profitable customers or zero budget",
                total_spend=0.0,
                expected_incremental_revenue=0.0,
                allocations=allocations,
                metadata={
                    "solver": "GreedyROISolver",
                    "targeted_count": 0,
                    "expected_net_profit": 0.0,
                },
            )

        safe_costs = np.where(tc[eligible_indices] > 0, tc[eligible_indices], 1e-9)
        roi_scores = feasible.net_expected_profits[eligible_indices] / safe_costs
        sorted_order = eligible_indices[np.argsort(-roi_scores)]

        total_spend = 0.0
        for idx in sorted_order:
            cost = float(tc[idx])
            if total_spend + cost <= self.budget:
                allocations[idx] = 1
                total_spend += cost

        inc_rev = float(np.sum((te * cv) * allocations))
        net_profit = inc_rev - total_spend

        return OptimizationResult(
            success=True,
            status_message="Greedy ROI allocation completed",
            total_spend=round(total_spend, 4),
            expected_incremental_revenue=round(inc_rev, 4),
            allocations=allocations,
            metadata={
                "solver": "GreedyROISolver",
                "targeted_count": int(np.sum(allocations)),
                "expected_net_profit": round(net_profit, 4),
            },
        )