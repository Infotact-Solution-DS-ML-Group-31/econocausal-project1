from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


REQUIRED_COLUMNS = {
    "customer_id",
    "offer_id",
    "cost",
    "expected_incremental_value",
}


@dataclass
class AllocationResult:
    allocations: pd.DataFrame
    budget: float
    total_cost: float
    total_expected_value: float
    selected_count: int
    solver_status: str


def optimize_allocation(
    offers: pd.DataFrame,
    budget: float,
) -> AllocationResult:
    """
    Select offers to maximize expected incremental value.

    Each row represents one offer option for one customer.
    Each customer can receive at most one selected offer.

    expected_incremental_value must be supplied by a separately
    validated predictive/causal model. This function does not
    estimate uplift or establish causality.
    """
    if not isinstance(offers, pd.DataFrame):
        raise TypeError("offers must be a pandas DataFrame.")

    if offers.empty:
        raise ValueError("offers dataset must not be empty.")

    missing = REQUIRED_COLUMNS - set(offers.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite, non-negative number.")

    df = offers.copy()

    for column in ["customer_id", "offer_id"]:
        if df[column].isna().any():
            raise ValueError(f"{column} must not contain missing values.")

    if df["offer_id"].duplicated().any():
        raise ValueError("offer_id values must be unique.")

    for column in ["cost", "expected_incremental_value"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite numeric values.")

    if (df["cost"] <= 0).any():
        raise ValueError("Every offer cost must be greater than zero.")

    n = len(df)
    customers = df["customer_id"].unique()

    # Objective is minimized, so negate expected value to maximize it.
    objective = -df["expected_incremental_value"].to_numpy(dtype=float)

    # First constraint: total offer cost <= budget.
    # Remaining constraints: at most one offer per customer.
    matrix = lil_matrix((1 + len(customers), n), dtype=float)
    matrix[0, :] = df["cost"].to_numpy(dtype=float)

    for row, customer in enumerate(customers, start=1):
        customer_mask = df["customer_id"].to_numpy() == customer
        matrix[row, customer_mask] = 1.0

    lower = np.full(1 + len(customers), -np.inf)
    upper = np.ones(1 + len(customers))
    upper[0] = float(budget)

    constraints = LinearConstraint(matrix.tocsr(), lower, upper)

    result = milp(
        c=objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"disp": False},
    )

    if not result.success or result.x is None:
        raise RuntimeError(f"Optimization failed: {result.message}")

    df["selected"] = np.rint(result.x).astype(int)

    selected = df[df["selected"] == 1].copy()
    total_cost = float(selected["cost"].sum())
    total_value = float(selected["expected_incremental_value"].sum())

    return AllocationResult(
        allocations=df,
        budget=float(budget),
        total_cost=total_cost,
        total_expected_value=total_value,
        selected_count=len(selected),
        solver_status=str(result.message),
    )
