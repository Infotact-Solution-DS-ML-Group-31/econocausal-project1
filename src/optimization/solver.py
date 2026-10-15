"""Budget-constrained discount optimization solver using SciPy HiGHS linear programming."""

from typing import Any, Dict, Optional
import numpy as np
from scipy.optimize import linprog

from src.api.schemas import OptimizationObjective
from src.optimization.base import BaseAllocationSolver, OptimizationResult
from src.optimization.constraints import prepare_feasible_inputs


class SciPyConstrainedSolver(BaseAllocationSolver):
    """Solves causal discount allocation using SciPy HiGHS linear programming.

    Maximizes campaign objectives (profit, revenue, or conversions) subject to
    a hard total budget constraint and customer eligibility criteria.
    """

    def __init__(
        self,
        budget: float,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_PROFIT,
        min_spend_per_user: float = 0.0,
        integer_rounding: bool = True,
    ):
        super().__init__(budget=budget, min_spend_per_user=min_spend_per_user)
        if isinstance(objective, str):
            self.objective = OptimizationObjective(objective)
        else:
            self.objective = objective
        self.integer_rounding = integer_rounding

    def solve(
        self,
        treatment_effects: np.ndarray,
        treatment_costs: np.ndarray,
        customer_ids: Optional[np.ndarray] = None,
        customer_values: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Execute linear programming optimization over eligible customers.

        Args:
            treatment_effects: Array of predicted ITE (uplift delta) values.
            treatment_costs: Array of promotional discount costs per customer.
            customer_ids: Optional array of customer identifiers.
            customer_values: Array of customer baseline values (e.g., AOV or LTV).
                Defaults to $50.0 per customer if not provided.

        Returns:
            OptimizationResult containing allocation vector, financial metrics, and metadata.
        """
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
            require_positive_net_profit=(self.objective == OptimizationObjective.MAXIMIZE_PROFIT),
        )

        te = feasible.treatment_effects
        tc = feasible.treatment_costs
        cv = feasible.customer_values

        if te.ndim != 1:
            raise ValueError("SciPyConstrainedSolver currently supports 1D treatment vectors.")

        allocations = np.zeros(te.shape[0], dtype=np.int32 if self.integer_rounding else np.float64)
        eligible_idx = np.where(feasible.eligible_mask)[0]

        # Edge case 1: Zero eligible candidates or non-positive budget
        if eligible_idx.size == 0 or self.budget <= 0.0:
            return OptimizationResult(
                success=True,
                status_message="No eligible candidates or zero budget; zero allocations made",
                total_spend=0.0,
                expected_incremental_revenue=0.0,
                allocations=allocations,
                metadata={
                    "solver": "SciPyConstrainedSolver (HiGHS)",
                    "objective": self.objective.value,
                    "targeted_count": 0,
                    "expected_net_profit": 0.0,
                    "status": "No eligible candidates or zero budget; zero allocations made",
                },
            )

        # Edge case 2: Budget below minimum eligible treatment cost
        eligible_costs = tc[eligible_idx]
        if eligible_costs.size > 0 and self.budget < float(np.min(eligible_costs)):
            return OptimizationResult(
                success=True,
                status_message="Budget below minimum treatment cost; zero allocations made",
                total_spend=0.0,
                expected_incremental_revenue=0.0,
                allocations=allocations,
                metadata={
                    "solver": "SciPyConstrainedSolver (HiGHS)",
                    "objective": self.objective.value,
                    "targeted_count": 0,
                    "expected_net_profit": 0.0,
                    "status": "Budget below minimum treatment cost; zero allocations made",
                },
            )

        # Formulate objective weights w_i
        if self.objective == OptimizationObjective.MAXIMIZE_PROFIT:
            weights = feasible.net_expected_profits[eligible_idx]
        elif self.objective == OptimizationObjective.MAXIMIZE_REVENUE:
            weights = (te * cv)[eligible_idx]
        elif self.objective == OptimizationObjective.MAXIMIZE_CONVERSIONS:
            weights = te[eligible_idx]
        else:
            raise ValueError(f"Unsupported optimization objective: {self.objective}")

        # Linear programming setup: min c^T x s.t. A_ub x <= b_ub, 0 <= x <= 1
        c = -weights
        A_ub = tc[eligible_idx].reshape(1, -1)
        b_ub = [self.budget]
        bounds = [(0.0, 1.0)]

        opt_res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not opt_res.success:
            return OptimizationResult(
                success=False,
                status_message=f"LP optimization failed: {opt_res.message}",
                total_spend=0.0,
                expected_incremental_revenue=0.0,
                allocations=allocations,
                metadata={
                    "solver": "SciPyConstrainedSolver (HiGHS)",
                    "objective": self.objective.value,
                    "targeted_count": 0,
                    "expected_net_profit": 0.0,
                    "status": opt_res.message,
                },
            )

        x_sol = opt_res.x
        safe_costs = np.where(eligible_costs > 0, eligible_costs, 1e-9)
        efficiency = weights / safe_costs

        if self.integer_rounding:
            pos_mask = x_sol > 1e-6
            if np.any(pos_mask):
                pos_indices = np.where(pos_mask)[0]
                # Sort positive LP solutions by (-x_sol, -efficiency)
                sorted_indices = sorted(
                    pos_indices,
                    key=lambda i: (-float(x_sol[i]), -float(efficiency[i])),
                )
                total_spend = 0.0
                for i in sorted_indices:
                    orig_idx = eligible_idx[i]
                    cost = float(tc[orig_idx])
                    if total_spend + cost <= self.budget:
                        allocations[orig_idx] = 1
                        total_spend += cost
            else:
                total_spend = 0.0
        else:
            allocations[eligible_idx] = x_sol
            total_spend = float(np.sum(allocations * tc))

        inc_rev = float(np.sum((te * cv) * allocations))
        net_profit = inc_rev - total_spend

        return OptimizationResult(
            success=True,
            status_message="Constrained LP optimization completed",
            total_spend=round(total_spend, 4),
            expected_incremental_revenue=round(inc_rev, 4),
            allocations=allocations,
            metadata={
                "solver": "SciPyConstrainedSolver (HiGHS)",
                "objective": self.objective.value,
                "targeted_count": int(np.sum(allocations > 0)),
                "expected_net_profit": round(net_profit, 4),
            },
        )
