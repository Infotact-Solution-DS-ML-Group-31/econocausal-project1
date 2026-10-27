"""Pydantic data contracts for EconoCausal API and optimization pipelines."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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


class MultiTierCustomerDataPoint(BaseModel):
    """Customer-level multi-treatment effect scores across discount tiers."""
    customer_id: str = Field(..., description="Unique customer identifier")
    baseline_conversion_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted baseline conversion without discount"
    )
    tier_ites: List[float] = Field(
        ..., min_length=1, description="Predicted ITE for each discount tier (e.g., [$10_ite, $20_ite])"
    )
    customer_value: float = Field(
        default=50.0, gt=0.0, description="Expected average order value or customer lifetime value"
    )


class MultiTierOptimizationRequest(BaseModel):
    """Payload submitted to allocate budget across multiple discount tiers."""
    budget: float = Field(..., gt=0.0, description="Total available campaign budget")
    tier_costs: List[float] = Field(
        ..., min_length=1, description="Cost for each discount tier (e.g., [10.0, 20.0])"
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MAXIMIZE_PROFIT,
        description="Target business objective",
    )
    min_spend_per_user: float = Field(
        default=0.0, ge=0.0, description="Minimum subsidy per targeted user"
    )
    customers: List[MultiTierCustomerDataPoint] = Field(
        ..., min_length=1, description="List of customers with multi-tier ITE scores"
    )

    @field_validator("customers")
    @classmethod
    def validate_unique_customer_ids(
        cls, v: List[MultiTierCustomerDataPoint]
    ) -> List[MultiTierCustomerDataPoint]:
        ids = [c.customer_id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate customer_id detected in customer payload.")
        return v

    @model_validator(mode="after")
    def validate_tier_lengths(self) -> "MultiTierOptimizationRequest":
        num_tiers = len(self.tier_costs)
        for c in self.customers:
            if len(c.tier_ites) != num_tiers:
                raise ValueError(
                    f"Customer '{c.customer_id}' has {len(c.tier_ites)} tier ITEs, "
                    f"expected {num_tiers} to match tier_costs."
                )
        return self


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


class IngestAndAllocateRequest(BaseModel):
    """Payload to ingest raw M2 causal inference records and perform budget allocation."""
    records: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="Raw customer/treatment records from causal ML pipeline"
    )
    budget: float = Field(..., gt=0.0, description="Total available campaign budget")
    default_cost: float = Field(
        default=10.0, gt=0.0, description="Default promotional cost if missing from records"
    )
    default_value: float = Field(
        default=50.0, gt=0.0, description="Default customer economic value if missing"
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MAXIMIZE_PROFIT,
        description="Target business objective",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "records" not in data and "customers" in data:
                data["records"] = data["customers"]
            elif "records" not in data and "data" in data:
                data["records"] = data["data"]
        return data


class UpliftDecilePoint(BaseModel):
    """Cumulative performance metrics for a specific population percentile bucket."""
    percentile: float = Field(..., description="Population percentile (e.g., 10.0, 20.0, ..., 100.0)")
    customer_count: int = Field(..., description="Cumulative number of customers targeted")
    cumulative_spend: float = Field(..., description="Cumulative treatment cost spend")
    cumulative_incremental_conversions: float = Field(
        ..., description="Cumulative expected incremental conversions"
    )
    cumulative_incremental_revenue: float = Field(
        ..., description="Cumulative expected incremental revenue"
    )
    cumulative_net_profit: float = Field(..., description="Cumulative net profit (revenue - spend)")
    random_baseline_revenue: float = Field(
        ..., description="Expected cumulative incremental revenue under random rollout"
    )


class UpliftCurveRequest(BaseModel):
    """Payload to compute cumulative uplift deciles, AUUC, and Qini metrics."""
    customers: List[CustomerDataPoint] = Field(
        ..., min_length=1, description="List of customers with ITE scores"
    )
    num_bins: int = Field(
        default=10, ge=2, le=50, description="Number of population percentile bins (e.g., 10 for deciles)"
    )

    @field_validator("customers")
    @classmethod
    def validate_unique_customer_ids(cls, v: List[CustomerDataPoint]) -> List[CustomerDataPoint]:
        ids = [c.customer_id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate customer_id detected in customer payload.")
        return v


class UpliftCurveResponse(BaseModel):
    """Cumulative uplift curve coordinates, AUUC score, and Qini evaluation report."""
    deciles: List[UpliftDecilePoint] = Field(
        ..., description="Percentile bucket points for Plotly.js curve visualization"
    )
    auuc_score: float = Field(..., description="Area Under the Uplift Curve")
    qini_coefficient: float = Field(
        ..., description="Normalized area between causal uplift curve and random baseline"
    )
    max_profit_percentile: float = Field(
        ..., description="Population percentage that maximizes cumulative net profit"
    )
    total_customers: int = Field(..., description="Total evaluated customer count")

class SegmentSummarySchema(BaseModel):
    """Causal uplift quadrant summary metrics for campaign decisioning."""
    segment_name: str = Field(
        ..., description="Quadrant name ('Persuadables', 'Sure Things', 'Lost Causes', or 'Sleeping Dogs')"
    )
    customer_count: int = Field(..., ge=0, description="Number of customers classified in this quadrant")
    population_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of total cohort population")
    mean_ite: float = Field(..., description="Average Individual Treatment Effect for this quadrant")
    mean_baseline_prob: float = Field(..., ge=0.0, le=1.0, description="Average baseline conversion probability")
    recommended_action: str = Field(..., description="Recommended marketing / discount action for this segment")


class ExecutiveReportResponse(BaseModel):
    """High-level executive summary report comparing causal optimization to blanket discounting."""
    total_customers: int = Field(..., ge=0, description="Total customers evaluated in campaign cohort")
    budget: float = Field(..., ge=0.0, description="Total campaign budget")
    total_spend: float = Field(..., ge=0.0, description="Total budget spent under causal optimization")
    expected_incremental_revenue: float = Field(..., description="Total expected incremental revenue generated")
    expected_net_profit: float = Field(..., description="Total expected net profit (revenue - spend)")
    roi_pct: float = Field(..., description="Net campaign Return on Investment percentage")
    savings_vs_blanket: float = Field(..., description="Net profit improvement of causal SciPy over blanket baseline")
    segments: List[SegmentSummarySchema] = Field(..., description="Quadrant breakdown of customer cohort")
    headline_summary: str = Field(..., description="Natural language executive summary headline")
    bullet_insights: List[str] = Field(..., description="Key actionable business insights")

