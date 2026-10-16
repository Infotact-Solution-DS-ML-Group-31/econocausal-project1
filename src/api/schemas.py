"""Pydantic data contracts for EconoCausal API and optimization pipelines."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class OptimizationObjective(str, Enum):
    """Supported business objectives for discount allocation."""
    MAXIMIZE_REVENUE = "maximize_revenue"
    MAXIMIZE_PROFIT = "maximize_profit"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"


class CustomerDataPoint(BaseModel):
    """Customer-level treatment effect score produced by M2 (Causal ML)."""
    customer_id: str = Field(..., description="Unique customer identifier")
    baseline_conversion_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted baseline conversion without discount"
    )
    predicted_ite: float = Field(
        ..., description="Estimated Individual Treatment Effect (uplift delta)"
    )
    treatment_cost: float = Field(
        ..., ge=0.0, description="Promotional cost of the discount tier (e.g., $10 or $20)"
    )
    customer_value: float = Field(
        default=50.0, gt=0.0, description="Expected average order value or customer lifetime value"
    )


class OptimizationRequest(BaseModel):
    """Payload submitted by M4 (UI) to run budget-constrained allocation."""
    budget: float = Field(..., gt=0.0, description="Total available campaign budget")
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MAXIMIZE_PROFIT,
        description="Target business objective",
    )
    min_spend_per_user: float = Field(
        default=0.0, ge=0.0, description="Minimum subsidy per targeted user"
    )
    customers: List[CustomerDataPoint] = Field(
        ..., min_length=1, description="List of customers with ITE scores"
    )

    @field_validator("customers")
    @classmethod
    def validate_unique_customer_ids(cls, v: List[CustomerDataPoint]) -> List[CustomerDataPoint]:
        ids = [c.customer_id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate customer_id detected in customer payload.")
        return v


class CustomerAllocation(BaseModel):
    """Recommended discount decision for an individual customer."""
    customer_id: str
    assigned_discount: float = Field(..., description="Discount allocated ($0, $10, $20, etc.)")
    is_targeted: bool = Field(..., description="Whether customer received a promotional offer")
    expected_incremental_revenue: float
    expected_net_profit: float


class OptimizationResponse(BaseModel):
    """Response returned to M4 for dashboard metrics and table rendering."""
    success: bool
    status_message: str
    total_budget: float
    total_spend: float
    budget_utilization_pct: float
    total_expected_incremental_revenue: float
    total_expected_net_profit: float
    targeted_customers_count: int
    total_customers_count: int
    allocations: List[CustomerAllocation]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyComparisonResponse(BaseModel):
    """Response comparing causal optimization policy against blanket discount baseline."""
    budget: float
    causal_scipy_policy: OptimizationResponse
    blanket_baseline_policy: OptimizationResponse
    net_profit_improvement: float
    executive_summary: str


class DriftCheckRequest(BaseModel):
    """Payload to evaluate distribution drift between reference and current ITE scores."""
    reference_ites: List[float] = Field(
        ..., min_length=5, description="Historical reference treatment effect distribution"
    )
    current_ites: List[float] = Field(
        ..., min_length=5, description="Current scoring batch treatment effect distribution"
    )
    psi_threshold: float = Field(
        default=0.25, gt=0.0, description="PSI threshold for critical distribution drift"
    )
    ks_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Significance level alpha for Kolmogorov-Smirnov test"
    )


class DriftCheckResponse(BaseModel):
    """Statistical distribution drift evaluation report."""
    drift_detected: bool
    severity: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    recommendation: str
    reference_count: int
    current_count: int

