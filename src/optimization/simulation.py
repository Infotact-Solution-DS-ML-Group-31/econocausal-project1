"""Monte Carlo policy confidence simulation for causal discount optimization."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.api.schemas import OptimizationRequest
from src.optimization.solver import SciPyConstrainedSolver


@dataclass(frozen=True)
class MonteCarloSimulationResult:
    """Outcomes of Monte Carlo simulations evaluating policy risk under ITE uncertainty."""

    num_simulations: int
    budget: float
    total_spend: float
    mean_net_profit: float
    std_net_profit: float
    ci_lower_5pct: float
    ci_upper_95pct: float
    probability_of_positive_roi: float


def run_policy_monte_carlo(
    request: OptimizationRequest,
    ite_std_error: float = 0.03,
    num_simulations: int = 200,
    seed: int = 42,
) -> MonteCarloSimulationResult:
    """Simulate policy performance distribution across perturbed treatment effect realizations.

    Parameters:
        request: OptimizationRequest defining budget, objective, and customer cohort.
        ite_std_error: Standard deviation of normal perturbation on predicted ITE scores.
        num_simulations: Number of Monte Carlo simulation trials to execute.
        seed: Random seed for reproducible sampling.

    Returns:
        MonteCarloSimulationResult with empirical distribution metrics and confidence intervals.
    """
    if ite_std_error < 0.0:
        raise ValueError(f"ite_std_error must be non-negative, got: {ite_std_error}")

    if not isinstance(num_simulations, int) or num_simulations < 1:
        raise ValueError(f"num_simulations must be an integer >= 1, got: {num_simulations}")

    if len(request.customers) == 0:
        raise ValueError("OptimizationRequest must contain at least one customer.")

    ite_vector = np.array([c.predicted_ite for c in request.customers], dtype=np.float64)
    cost_vector = np.array([c.treatment_cost for c in request.customers], dtype=np.float64)
    value_vector = np.array([c.customer_value for c in request.customers], dtype=np.float64)
    customer_ids = np.array([c.customer_id for c in request.customers])

    # 1. Determine optimal baseline allocation using SciPy solver
    solver = SciPyConstrainedSolver(
        budget=request.budget,
        objective=request.objective,
        min_spend_per_user=request.min_spend_per_user,
    )
    opt_result = solver.solve(
        treatment_effects=ite_vector,
        treatment_costs=cost_vector,
        customer_ids=customer_ids,
        customer_values=value_vector,
    )

    allocations = opt_result.allocations
    total_spend = round(float(opt_result.total_spend), 4)
    n_customers = len(request.customers)

    # 2. Draw perturbed ITE realizations: (num_simulations, N)
    rng = np.random.default_rng(seed)
    if ite_std_error == 0.0:
        perturbed_ites = np.tile(ite_vector, (num_simulations, 1))
    else:
        perturbed_ites = rng.normal(loc=ite_vector, scale=ite_std_error, size=(num_simulations, n_customers))

    # 3. Compute realized net profit for each simulation trial
    if allocations.ndim == 1:
        realized_inc_rev = np.sum((perturbed_ites * value_vector) * allocations, axis=1)
    else:
        realized_inc_rev = np.sum(perturbed_ites * value_vector[:, None] * allocations, axis=(1, 2))

    realized_net_profits = realized_inc_rev - total_spend

    # 4. Compute empirical distribution metrics
    mean_net_profit = round(float(np.mean(realized_net_profits)), 4)
    std_net_profit = round(float(np.std(realized_net_profits)), 4)
    ci_lower_5pct = round(float(np.percentile(realized_net_profits, 5.0)), 4)
    ci_upper_95pct = round(float(np.percentile(realized_net_profits, 95.0)), 4)

    # Empirical probability of positive net profit (ROI)
    probability_of_positive_roi = round(float(np.mean(realized_net_profits > 0.0)), 4)

    # Guard for exact precision when ite_std_error is 0.0
    if ite_std_error == 0.0:
        std_net_profit = 0.0
        ci_lower_5pct = mean_net_profit
        ci_upper_95pct = mean_net_profit

    return MonteCarloSimulationResult(
        num_simulations=num_simulations,
        budget=round(float(request.budget), 4),
        total_spend=total_spend,
        mean_net_profit=mean_net_profit,
        std_net_profit=std_net_profit,
        ci_lower_5pct=ci_lower_5pct,
        ci_upper_95pct=ci_upper_95pct,
        probability_of_positive_roi=probability_of_positive_roi,
    )

