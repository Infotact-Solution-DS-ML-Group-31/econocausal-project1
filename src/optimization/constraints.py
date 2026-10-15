"""Boundary checks, feasibility validation, and pre-filtering for budget optimization."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ValidatedSolverInputs:
    """Sanitized and validated arrays ready for mathematical optimization."""

    treatment_effects: np.ndarray
    treatment_costs: np.ndarray
    customer_values: np.ndarray
    net_expected_profits: np.ndarray
    eligible_mask: np.ndarray
    budget: float


def validate_solver_inputs(
    treatment_effects: np.ndarray,
    treatment_costs: np.ndarray,
    budget: float,
    customer_values: Optional[np.ndarray] = None,
    max_budget_limit: float = 1_000_000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate dimensionality and sanity of inputs prior to optimization."""
    if np.isnan(budget) or np.isinf(budget):
        raise ValueError("Budget cannot be NaN or Infinite.")
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got: {budget}")
    if budget > max_budget_limit:
        raise ValueError(f"Budget {budget} exceeds maximum allowed limit of {max_budget_limit}.")

    te = np.asarray(treatment_effects, dtype=np.float64)
    tc = np.asarray(treatment_costs, dtype=np.float64)

    if te.size == 0:
        raise ValueError("Treatment effects array cannot be empty.")
    if tc.size == 0:
        raise ValueError("Treatment costs array cannot be empty.")

    if np.any(np.isnan(te)) or np.any(np.isinf(te)):
        raise ValueError("Treatment effects contain NaN or Inf values.")
    if np.any(np.isnan(tc)) or np.any(np.isinf(tc)):
        raise ValueError("Treatment costs contain NaN or Inf values.")

    if np.any(tc < 0):
        raise ValueError("Treatment costs cannot be negative.")

    if te.ndim == 1:
        if tc.ndim == 1 and tc.shape[0] != te.shape[0]:
            raise ValueError(
                f"Shape mismatch: treatment_effects shape {te.shape} != treatment_costs shape {tc.shape}"
            )
    elif te.ndim == 2:
        if tc.ndim == 1 and tc.shape[0] != te.shape[1]:
            raise ValueError(
                f"2D shape mismatch: treatment_effects has {te.shape[1]} tiers, but treatment_costs has {tc.shape[0]} elements."
            )
        if tc.ndim == 2 and tc.shape != te.shape:
            raise ValueError(
                f"2D shape mismatch: treatment_effects shape {te.shape} != treatment_costs shape {tc.shape}"
            )
    else:
        raise ValueError(f"Unsupported treatment_effects dimensionality: {te.ndim}D (expected 1D or 2D).")

    if customer_values is not None:
        cv = np.asarray(customer_values, dtype=np.float64)
        if np.any(np.isnan(cv)) or np.any(np.isinf(cv)):
            raise ValueError("Customer values contain NaN or Inf values.")
        if np.any(cv <= 0):
            raise ValueError("Customer values must be strictly positive.")
        if cv.shape[0] != te.shape[0]:
            raise ValueError(
                f"Customer values length {cv.shape[0]} does not match customer count {te.shape[0]}."
            )

    return te, tc


def prepare_feasible_inputs(
    treatment_effects: np.ndarray,
    treatment_costs: np.ndarray,
    customer_values: np.ndarray,
    budget: float,
    require_positive_net_profit: bool = True,
) -> ValidatedSolverInputs:
    """Validate inputs and compute net profit vectors and eligibility masks."""
    te, tc = validate_solver_inputs(
        treatment_effects=treatment_effects,
        treatment_costs=treatment_costs,
        budget=budget,
        customer_values=customer_values,
    )
    cv = np.asarray(customer_values, dtype=np.float64)

    if te.ndim == 2 and cv.ndim == 1:
        inc_rev = te * cv[:, np.newaxis]
    else:
        inc_rev = te * cv

    net_profits = inc_rev - tc

    if require_positive_net_profit:
        eligible_mask = (te > 0) & (net_profits > 0)
    else:
        eligible_mask = te > 0

    return ValidatedSolverInputs(
        treatment_effects=te,
        treatment_costs=tc,
        customer_values=cv,
        net_expected_profits=net_profits,
        eligible_mask=eligible_mask,
        budget=float(budget),
    )