"""Causal uplift curve and Qini evaluation metrics for prescriptive targeting."""

from dataclasses import dataclass
from typing import List
import numpy as np

try:
    from scipy.integrate import trapezoid
except ImportError:
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from src.optimization.constraints import validate_solver_inputs


@dataclass(frozen=True)
class UpliftDecilePoint:
    """Cumulative performance metrics for a specific population percentile bucket."""

    percentile: float
    customer_count: int
    cumulative_spend: float
    cumulative_incremental_conversions: float
    cumulative_incremental_revenue: float
    cumulative_net_profit: float
    random_baseline_revenue: float


@dataclass(frozen=True)
class UpliftCurveReport:
    """Summary report detailing the cumulative uplift curve, AUUC, and Qini metrics."""

    deciles: List[UpliftDecilePoint]
    auuc_score: float
    qini_coefficient: float
    max_profit_percentile: float


def compute_uplift_curve(
    treatment_effects: np.ndarray,
    treatment_costs: np.ndarray,
    customer_values: np.ndarray,
    num_bins: int = 10,
) -> UpliftCurveReport:
    """Compute cumulative uplift deciles, AUUC score, and Qini coefficient.

    Parameters:
        treatment_effects: 1D array of predicted Individual Treatment Effects (uplift).
        treatment_costs: 1D array of promotional costs per customer.
        customer_values: 1D array of customer economic values (CLV or AOV).
        num_bins: Number of equal-sized population bins (default 10).

    Returns:
        UpliftCurveReport containing decile points, AUUC, Qini, and optimal rollout percentage.
    """
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError(f"num_bins must be an integer >= 2, got: {num_bins}")

    if customer_values is None:
        raise ValueError("Customer values array cannot be None.")

    # Validate inputs using standard solver input validation
    te, tc = validate_solver_inputs(
        treatment_effects=treatment_effects,
        treatment_costs=treatment_costs,
        budget=1_000_000.0,
        customer_values=customer_values,
    )

    if te.ndim != 1:
        raise ValueError(
            f"compute_uplift_curve only supports 1D treatment effects, got {te.ndim}D array."
        )

    cv = np.asarray(customer_values, dtype=np.float64)
    if cv.ndim != 1 or cv.shape[0] != te.shape[0]:
        raise ValueError(
            f"Shape mismatch: customer_values shape {cv.shape} != treatment_effects shape {te.shape}"
        )

    n_customers = te.shape[0]

    # Calculate expected net profit: ITE * value - cost
    net_profits = (te * cv) - tc

    # Sort customers in descending order of net expected profit
    sort_order = np.argsort(-net_profits, kind="mergesort")
    sorted_te = te[sort_order]
    sorted_tc = tc[sort_order]
    sorted_cv = cv[sort_order]
    sorted_rev = (te * cv)[sort_order]
    sorted_profit = net_profits[sort_order]

    total_incremental_revenue = float(np.sum(sorted_rev))

    deciles: List[UpliftDecilePoint] = []

    for b in range(1, num_bins + 1):
        percentile = round((b / num_bins) * 100.0, 2)
        k = int(np.round(b * n_customers / num_bins))
        k = min(n_customers, max(1, k))

        cum_spend = float(np.sum(sorted_tc[:k]))
        cum_conversions = float(np.sum(sorted_te[:k]))
        cum_revenue = float(np.sum(sorted_rev[:k]))
        cum_profit = float(np.sum(sorted_profit[:k]))
        random_baseline_rev = float(total_incremental_revenue * (percentile / 100.0))

        deciles.append(
            UpliftDecilePoint(
                percentile=percentile,
                customer_count=k,
                cumulative_spend=round(cum_spend, 4),
                cumulative_incremental_conversions=round(cum_conversions, 4),
                cumulative_incremental_revenue=round(cum_revenue, 4),
                cumulative_net_profit=round(cum_profit, 4),
                random_baseline_revenue=round(random_baseline_rev, 4),
            )
        )

    # Compute AUUC and Qini
    x_coords = np.array([0.0] + [d.percentile / 100.0 for d in deciles], dtype=np.float64)
    y_coords = np.array([0.0] + [d.cumulative_incremental_revenue for d in deciles], dtype=np.float64)
    y_rand_coords = np.array([0.0] + [d.random_baseline_revenue for d in deciles], dtype=np.float64)

    auuc_score = float(trapezoid(y_coords, x_coords))
    area_random = float(trapezoid(y_rand_coords, x_coords))

    if abs(area_random) > 1e-9:
        qini_coefficient = float((auuc_score - area_random) / abs(area_random))
    else:
        qini_coefficient = float(auuc_score)

    # Find population percentile that maximizes cumulative net profit
    best_point = max(deciles, key=lambda d: d.cumulative_net_profit)
    max_profit_percentile = float(best_point.percentile)

    return UpliftCurveReport(
        deciles=deciles,
        auuc_score=round(auuc_score, 4),
        qini_coefficient=round(qini_coefficient, 4),
        max_profit_percentile=round(max_profit_percentile, 2),
    )
