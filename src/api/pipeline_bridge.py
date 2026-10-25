"""Pipeline bridge module connecting M2 (Causal ML) inference outputs with M3 (Optimization).

Provides ingestion adapters for tabular EconML/DoWhy causal model outputs,
flexible field aliasing, probability boundary clamping, discount allocation execution,
and export serializers for M4 (UI) and CSV reports.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.api.schemas import (
    CustomerAllocation,
    CustomerDataPoint,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResponse,
)
from src.optimization.base import OptimizationResult
from src.optimization.solver import SciPyConstrainedSolver

# Field alias lookups for tabular causal ML outputs
_CUSTOMER_ID_ALIASES = ("customer_id", "id", "user_id", "client_id", "cust_id", "uuid")
_ITE_ALIASES = ("predicted_ite", "ite", "cate", "tau", "treatment_effect", "uplift", "effect")
_PROB_ALIASES = (
    "baseline_conversion_prob",
    "prob",
    "baseline_prob",
    "conversion_prob",
    "base_prob",
    "propensity",
    "y0",
)
_COST_ALIASES = ("treatment_cost", "cost", "discount", "discount_cost", "treatment_spend")
_VALUE_ALIASES = ("customer_value", "value", "aov", "clv", "revenue", "ltv", "customer_ltv")


def _extract_field(record: Dict[str, Any], aliases: tuple, default: Any = None) -> Any:
    """Helper to extract a value from record using a sequence of alias candidate keys."""
    for key in aliases:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _build_optimization_response(
    request: OptimizationRequest,
    result: OptimizationResult,
) -> OptimizationResponse:
    """Helper to convert an OptimizationResult and request into an OptimizationResponse."""
    ite_vector = np.array([c.predicted_ite for c in request.customers], dtype=np.float64)
    cost_vector = np.array([c.treatment_cost for c in request.customers], dtype=np.float64)
    value_vector = np.array([c.customer_value for c in request.customers], dtype=np.float64)

    customer_allocations: List[CustomerAllocation] = []
    for i, c in enumerate(request.customers):
        alloc_val = float(result.allocations[i])
        is_targeted = bool(alloc_val > 0)
        assigned_discount = float(cost_vector[i] * alloc_val)
        exp_inc_rev = float((ite_vector[i] * value_vector[i]) * alloc_val)
        exp_net_profit = float(exp_inc_rev - assigned_discount)

        customer_allocations.append(
            CustomerAllocation(
                customer_id=c.customer_id,
                assigned_discount=round(assigned_discount, 4),
                is_targeted=is_targeted,
                expected_incremental_revenue=round(exp_inc_rev, 4),
                expected_net_profit=round(exp_net_profit, 4),
            )
        )

    budget_util_pct = (
        round((result.total_spend / request.budget) * 100.0, 2)
        if request.budget > 0
        else 0.0
    )
    net_profit = float(
        result.metadata.get(
            "expected_net_profit",
            result.expected_incremental_revenue - result.total_spend,
        )
    )

    return OptimizationResponse(
        success=result.success,
        status_message=result.status_message,
        total_budget=round(float(request.budget), 4),
        total_spend=round(float(result.total_spend), 4),
        budget_utilization_pct=budget_util_pct,
        total_expected_incremental_revenue=round(float(result.expected_incremental_revenue), 4),
        total_expected_net_profit=round(net_profit, 4),
        targeted_customers_count=int(
            result.metadata.get("targeted_count", np.sum(result.allocations > 0))
        ),
        total_customers_count=len(request.customers),
        allocations=customer_allocations,
        metadata=result.metadata,
    )


def build_request_from_causal_records(
    records: List[Dict[str, Any]],
    budget: float,
    default_cost: float = 10.0,
    default_value: float = 50.0,
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_PROFIT,
) -> OptimizationRequest:
    """Convert raw dictionary records emitted by M2's Causal ML pipeline into an OptimizationRequest.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        List of customer records with heterogeneous treatment effect and propensity keys.
    budget : float
        Available campaign budget ceiling (must be > 0.0).
    default_cost : float, default=10.0
        Fallback promotional subsidy cost when unspecified in raw records.
    default_value : float, default=50.0
        Fallback customer order/lifetime value when unspecified in raw records.
    objective : OptimizationObjective, default=OptimizationObjective.MAXIMIZE_PROFIT
        Target business objective for the LP solver.

    Returns
    -------
    OptimizationRequest
        Validated Pydantic data contract ready for budget optimization.
    """
    if not records:
        raise ValueError("records list cannot be empty.")
    if budget <= 0.0:
        raise ValueError("budget must be greater than 0.0.")

    customers: List[CustomerDataPoint] = []

    for idx, r in enumerate(records):
        # 1. Customer ID
        cid_val = _extract_field(r, _CUSTOMER_ID_ALIASES)
        customer_id = str(cid_val) if cid_val is not None else f"cust_{idx + 1}"

        # 2. Predicted ITE / CATE
        ite_raw = _extract_field(r, _ITE_ALIASES, default=0.0)
        try:
            ite = float(ite_raw)
        except (ValueError, TypeError):
            ite = 0.0

        # 3. Baseline conversion probability (clamped into [0.0, 1.0])
        prob_raw = _extract_field(r, _PROB_ALIASES, default=0.10)
        try:
            prob = float(prob_raw)
        except (ValueError, TypeError):
            prob = 0.10
        clamped_prob = max(0.0, min(1.0, prob))

        # 4. Treatment cost
        cost_raw = _extract_field(r, _COST_ALIASES, default=default_cost)
        try:
            cost = float(cost_raw)
        except (ValueError, TypeError):
            cost = float(default_cost)
        cost = max(0.0, cost)

        # 5. Customer value (AOV / CLV)
        val_raw = _extract_field(r, _VALUE_ALIASES, default=default_value)
        try:
            val = float(val_raw)
        except (ValueError, TypeError):
            val = float(default_value)
        val = max(0.01, val)

        customers.append(
            CustomerDataPoint(
                customer_id=customer_id,
                baseline_conversion_prob=clamped_prob,
                predicted_ite=ite,
                treatment_cost=cost,
                customer_value=val,
            )
        )

    return OptimizationRequest(
        budget=budget,
        objective=objective,
        customers=customers,
    )


def allocate_discounts(request: OptimizationRequest) -> OptimizationResponse:
    """Execute SciPy-constrained linear programming optimization for an OptimizationRequest.

    Parameters
    ----------
    request : OptimizationRequest
        Validated optimization request containing customers, budget, and business objective.

    Returns
    -------
    OptimizationResponse
        Optimization outcome including individual customer allocations and aggregate metrics.
    """
    customer_ids = np.array([c.customer_id for c in request.customers])
    ite_vector = np.array([c.predicted_ite for c in request.customers], dtype=np.float64)
    cost_vector = np.array([c.treatment_cost for c in request.customers], dtype=np.float64)
    value_vector = np.array([c.customer_value for c in request.customers], dtype=np.float64)

    solver = SciPyConstrainedSolver(
        budget=request.budget,
        objective=request.objective,
        min_spend_per_user=request.min_spend_per_user,
        integer_rounding=True,
    )
    result = solver.solve(
        treatment_effects=ite_vector,
        treatment_costs=cost_vector,
        customer_ids=customer_ids,
        customer_values=value_vector,
    )

    return _build_optimization_response(request, result)


def export_allocation_to_records(response: OptimizationResponse) -> List[Dict[str, Any]]:
    """Convert an OptimizationResponse into flat dictionary records suitable for CSV export or tabular UI.

    Parameters
    ----------
    response : OptimizationResponse
        Output produced by the optimization engine.

    Returns
    -------
    List[Dict[str, Any]]
        List of flat customer allocation records with ROI percentage calculation.
    """
    flat_records: List[Dict[str, Any]] = []

    for alloc in response.allocations:
        assigned_disc = float(alloc.assigned_discount)
        exp_inc_rev = float(alloc.expected_incremental_revenue)
        exp_net_profit = float(alloc.expected_net_profit)

        # Net ROI % = (expected_net_profit / assigned_discount) * 100
        if assigned_disc > 0.0:
            roi_pct = round((exp_net_profit / assigned_disc) * 100.0, 2)
        else:
            roi_pct = 0.0

        flat_records.append(
            {
                "customer_id": alloc.customer_id,
                "is_targeted": alloc.is_targeted,
                "assigned_discount": round(assigned_disc, 4),
                "expected_incremental_revenue": round(exp_inc_rev, 4),
                "expected_net_profit": round(exp_net_profit, 4),
                "roi_pct": roi_pct,
            }
        )

    return flat_records
