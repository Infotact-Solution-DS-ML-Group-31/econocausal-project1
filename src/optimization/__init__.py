"""Optimization package exposing solvers and constraint validators."""

from src.optimization.base import BaseAllocationSolver, OptimizationResult
from src.optimization.baseline import BlanketDiscountSolver, GreedyROISolver
from src.optimization.constraints import (
    ValidatedSolverInputs,
    prepare_feasible_inputs,
    validate_solver_inputs,
)
from src.optimization.drift import (
    DriftReport,
    calculate_psi,
    evaluate_distribution_drift,
)
from src.optimization.reporting import (
    ExecutiveCampaignReport,
    SegmentSummary,
    classify_customer_segments,
    generate_executive_report,
)
from src.optimization.simulation import (
    MonteCarloSimulationResult,
    run_policy_monte_carlo,
)
from src.optimization.solver import SciPyConstrainedSolver
from src.optimization.uplift_metrics import (
    UpliftCurveReport,
    UpliftDecilePoint,
    compute_uplift_curve,
)

__all__ = [
    "BaseAllocationSolver",
    "OptimizationResult",
    "BlanketDiscountSolver",
    "GreedyROISolver",
    "ValidatedSolverInputs",
    "prepare_feasible_inputs",
    "validate_solver_inputs",
    "SciPyConstrainedSolver",
    "DriftReport",
    "calculate_psi",
    "evaluate_distribution_drift",
    "UpliftCurveReport",
    "UpliftDecilePoint",
    "compute_uplift_curve",
    "SegmentSummary",
    "ExecutiveCampaignReport",
    "classify_customer_segments",
    "generate_executive_report",
    "MonteCarloSimulationResult",
    "run_policy_monte_carlo",
]

