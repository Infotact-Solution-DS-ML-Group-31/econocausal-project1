"""Optimization package exposing solvers and constraint validators."""

from src.optimization.base import BaseAllocationSolver, OptimizationResult
from src.optimization.baseline import BlanketDiscountSolver, GreedyROISolver
from src.optimization.constraints import (
    ValidatedSolverInputs,
    prepare_feasible_inputs,
    validate_solver_inputs,
)
from src.optimization.solver import SciPyConstrainedSolver

__all__ = [
    "BaseAllocationSolver",
    "OptimizationResult",
    "BlanketDiscountSolver",
    "GreedyROISolver",
    "ValidatedSolverInputs",
    "prepare_feasible_inputs",
    "validate_solver_inputs",
    "SciPyConstrainedSolver",
]
