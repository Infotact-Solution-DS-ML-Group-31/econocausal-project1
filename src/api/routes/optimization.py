"""API routes for budget-constrained discount optimization and policy comparison."""

from typing import Any, Dict, List
from fastapi import APIRouter, status
import numpy as np

from src.api.schemas import (
    CustomerAllocation,
    DriftCheckRequest,
    DriftCheckResponse,
    IngestAndAllocateRequest,
    MultiTierOptimizationRequest,
    OptimizationRequest,
    OptimizationResponse,
    PolicyComparisonResponse,
)
from src.api.pipeline_bridge import (
    allocate_discounts,
    build_request_from_causal_records,
    export_allocation_to_records,
)
from src.optimization.base import OptimizationResult
from src.optimization.baseline import BlanketDiscountSolver
from src.optimization.drift import evaluate_distribution_drift
from src.optimization.solver import SciPyConstrainedSolver

router = APIRouter(prefix="/api/v1", tags=["Optimization & Allocation"])


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


def _build_multi_tier_optimization_response(
    request: MultiTierOptimizationRequest,
    result: OptimizationResult,
    te: np.ndarray,
    tc: np.ndarray,
    cv: np.ndarray,
) -> OptimizationResponse:
    """Helper to convert an OptimizationResult and multi-tier request into an OptimizationResponse."""
    customer_allocations: List[CustomerAllocation] = []
    allocations_matrix = result.allocations  # shape (N, K)

    for i, c in enumerate(request.customers):
        row_alloc = allocations_matrix[i]
        selected_tier_indices = np.where(row_alloc > 0)[0]
        if len(selected_tier_indices) > 0:
            k = int(selected_tier_indices[0])
            alloc_val = float(row_alloc[k])
            is_targeted = True
            assigned_discount = float(tc[k] * alloc_val)
            exp_inc_rev = float((te[i, k] * cv[i]) * alloc_val)
            exp_net_profit = float(exp_inc_rev - assigned_discount)
        else:
            is_targeted = False
            assigned_discount = 0.0
            exp_inc_rev = 0.0
            exp_net_profit = 0.0

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
            result.metadata.get("targeted_count", np.sum(np.any(allocations_matrix > 0, axis=1)))
        ),
        total_customers_count=len(request.customers),
        allocations=customer_allocations,
        metadata=result.metadata,
    )


@router.post(
    "/allocate",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute optimal discount allocation under budget constraints",
)
async def allocate_budget(request: OptimizationRequest) -> OptimizationResponse:
    """Execute SciPy-constrained linear programming optimization for customer discount allocation."""
    return allocate_discounts(request)


@router.post(
    "/ingest-and-allocate",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest raw causal ML records from M2 and compute optimal discount allocation",
)
async def ingest_and_allocate(
    payload: IngestAndAllocateRequest,
) -> OptimizationResponse:
    """Bridge endpoint converting raw EconML/DoWhy records directly into optimal discount allocations."""
    opt_request = build_request_from_causal_records(
        records=payload.records,
        budget=payload.budget,
        default_cost=payload.default_cost,
        default_value=payload.default_value,
        objective=payload.objective,
    )
    return allocate_discounts(opt_request)


@router.post(
    "/allocate-multi-tier",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute optimal multi-tier discount allocation under budget constraints",
)
async def allocate_multi_tier_budget(
    request: MultiTierOptimizationRequest,
) -> OptimizationResponse:
    """Execute SciPy-constrained linear programming optimization for multi-tier discount allocation."""
    customer_ids = np.array([c.customer_id for c in request.customers])
    te = np.array([c.tier_ites for c in request.customers], dtype=np.float64)
    tc = np.array(request.tier_costs, dtype=np.float64)
    cv = np.array([c.customer_value for c in request.customers], dtype=np.float64)

    solver = SciPyConstrainedSolver(
        budget=request.budget,
        objective=request.objective,
        min_spend_per_user=request.min_spend_per_user,
        integer_rounding=True,
    )
    result = solver.solve(
        treatment_effects=te,
        treatment_costs=tc,
        customer_ids=customer_ids,
        customer_values=cv,
    )

    return _build_multi_tier_optimization_response(request, result, te, tc, cv)


@router.post(
    "/compare-policies",
    response_model=PolicyComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Benchmark causal optimization against naive blanket discount baseline",
)
async def compare_policies(request: OptimizationRequest) -> Dict[str, Any]:
    """Benchmark causal optimization against standard blanket discount allocation."""
    customer_ids = np.array([c.customer_id for c in request.customers])
    ite_vector = np.array([c.predicted_ite for c in request.customers], dtype=np.float64)
    cost_vector = np.array([c.treatment_cost for c in request.customers], dtype=np.float64)
    value_vector = np.array([c.customer_value for c in request.customers], dtype=np.float64)

    # 1. Causal SciPy Policy
    scipy_solver = SciPyConstrainedSolver(
        budget=request.budget,
        objective=request.objective,
        min_spend_per_user=request.min_spend_per_user,
        integer_rounding=True,
    )
    scipy_res = scipy_solver.solve(
        treatment_effects=ite_vector,
        treatment_costs=cost_vector,
        customer_ids=customer_ids,
        customer_values=value_vector,
    )
    causal_scipy_policy = _build_optimization_response(request, scipy_res)

    # 2. Blanket Baseline Policy
    blanket_solver = BlanketDiscountSolver(
        budget=request.budget,
        min_spend_per_user=request.min_spend_per_user,
    )
    blanket_res = blanket_solver.solve(
        treatment_effects=ite_vector,
        treatment_costs=cost_vector,
        customer_ids=customer_ids,
        customer_values=value_vector,
    )
    blanket_baseline_policy = _build_optimization_response(request, blanket_res)

    # 3. Delta & Summary
    net_profit_diff = (
        causal_scipy_policy.total_expected_net_profit
        - blanket_baseline_policy.total_expected_net_profit
    )
    net_profit_improvement = round(net_profit_diff, 4)

    summary = (
        f"Causal optimization improves net profit by ${net_profit_improvement:.2f} "
        f"compared to blanket targeting."
    )

    return {
        "budget": request.budget,
        "causal_scipy_policy": causal_scipy_policy,
        "blanket_baseline_policy": blanket_baseline_policy,
        "net_profit_improvement": net_profit_improvement,
        "executive_summary": summary,
    }


@router.post(
    "/drift-check",
    response_model=DriftCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate distribution drift between reference and current ITE distributions",
)
async def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Evaluate covariate and treatment effect distribution drift using PSI and KS-2sample test."""
    ref_arr = np.array(request.reference_ites, dtype=np.float64)
    cur_arr = np.array(request.current_ites, dtype=np.float64)

    report = evaluate_distribution_drift(
        reference=ref_arr,
        current=cur_arr,
        psi_threshold=request.psi_threshold,
        ks_alpha=request.ks_alpha,
    )

    return DriftCheckResponse(
        drift_detected=report.drift_detected,
        severity=report.severity,
        psi_score=report.psi_score,
        ks_statistic=report.ks_statistic,
        ks_pvalue=report.ks_pvalue,
        recommendation=report.recommendation,
        reference_count=len(request.reference_ites),
        current_count=len(request.current_ites),
    )
