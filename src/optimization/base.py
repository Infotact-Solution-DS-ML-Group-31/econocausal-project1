"""Base interfaces and data structures for budget-constrained discount optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable outcome of an allocation pass."""
    success: bool
    status_message: str
    total_spend: float
    expected_incremental_revenue: float
    allocations: np.ndarray  # Shape: (N, num_treatments) or binary assignment
    metadata: Dict[str, Any]


class BaseAllocationSolver(ABC):
    """Abstract base class for discount policy solvers."""

    def __init__(self, budget: float, min_spend_per_user: float = 0.0):
        if budget < 0:
            raise ValueError("Budget must be non-negative.")
        self.budget = budget
        self.min_spend_per_user = min_spend_per_user

    @abstractmethod
    def solve(
        self,
        treatment_effects: np.ndarray,
        treatment_costs: np.ndarray,
        customer_ids: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Solve the allocation problem maximizing causal uplift subject to budget constraints.

        Args:
            treatment_effects: Array of estimated ITE scores (uplift / conversion delta).
            treatment_costs: Direct promotional cost per treatment tier.
            customer_ids: Optional customer identifiers.

        Returns:
            OptimizationResult containing allocation matrix and metrics.
        """
        pass
    