"""Customer segment classifier and executive insight report generator."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.api.schemas import OptimizationRequest
from src.optimization.baseline import BlanketDiscountSolver
from src.optimization.solver import SciPyConstrainedSolver


@dataclass(frozen=True)
class SegmentSummary:
    """Aggregated metrics and recommendations for a single causal uplift quadrant."""

    segment_name: str
    customer_count: int
    population_pct: float
    mean_ite: float
    mean_baseline_prob: float
    recommended_action: str


@dataclass(frozen=True)
class ExecutiveCampaignReport:
    """Comprehensive executive insight report comparing causal optimization to baseline."""

    total_customers: int
    budget: float
    total_spend: float
    expected_incremental_revenue: float
    expected_net_profit: float
    roi_pct: float
    savings_vs_blanket: float
    segments: List[SegmentSummary]
    headline_summary: str
    bullet_insights: List[str]


def classify_customer_segments(
    baseline_probs: np.ndarray,
    treatment_effects: np.ndarray,
    treatment_costs: np.ndarray,
    customer_values: np.ndarray,
) -> List[SegmentSummary]:
    """Classify customers into the four classic causal uplift quadrants.

    Quadrants:
      - Sleeping Dogs: ITE < 0 (Do not discount; offer triggers negative response).
      - Persuadables: ITE > 0 and (ITE * value - cost) > 0 (Primary target for promotional spend).
      - Sure Things: Not in Persuadables/Sleeping Dogs and baseline_prob >= 0.50 (High baseline conversion; discount unnecessary).
      - Lost Causes: Remaining low-uplift, low-probability users (Holdout; subsidy does not cover cost).

    Parameters:
        baseline_probs: 1D array of baseline conversion probabilities.
        treatment_effects: 1D array of Individual Treatment Effects (uplift deltas).
        treatment_costs: 1D array of promotional costs per customer.
        customer_values: 1D array of customer economic values (AOV or CLV).

    Returns:
        List of SegmentSummary instances for each of the four quadrants.
    """
    probs = np.asarray(baseline_probs, dtype=np.float64)
    ites = np.asarray(treatment_effects, dtype=np.float64)
    costs = np.asarray(treatment_costs, dtype=np.float64)
    values = np.asarray(customer_values, dtype=np.float64)

    if probs.ndim != 1 or ites.ndim != 1 or costs.ndim != 1 or values.ndim != 1:
        raise ValueError("All input arrays must be 1-dimensional.")

    n_customers = probs.shape[0]
    if not (ites.shape[0] == n_customers and costs.shape[0] == n_customers and values.shape[0] == n_customers):
        raise ValueError(
            f"All input arrays must have the same length: probs={n_customers}, "
            f"ites={ites.shape[0]}, costs={costs.shape[0]}, values={values.shape[0]}."
        )

    if (
        np.isnan(probs).any()
        or np.isnan(ites).any()
        or np.isnan(costs).any()
        or np.isnan(values).any()
    ):
        raise ValueError("Input arrays must not contain NaN values.")

    if (
        np.isinf(probs).any()
        or np.isinf(ites).any()
        or np.isinf(costs).any()
        or np.isinf(values).any()
    ):
        raise ValueError("Input arrays must not contain infinite values.")

    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Baseline conversion probabilities must be within [0.0, 1.0].")

    if np.any(costs < 0.0):
        raise ValueError("Treatment costs must be non-negative.")

    if np.any(values < 0.0):
        raise ValueError("Customer values must be non-negative.")

    if n_customers == 0:
        return [
            SegmentSummary("Persuadables", 0, 0.0, 0.0, 0.0, "Primary target for promotional spend."),
            SegmentSummary("Sure Things", 0, 0.0, 0.0, 0.0, "High baseline conversion; discount unnecessary."),
            SegmentSummary("Lost Causes", 0, 0.0, 0.0, 0.0, "Holdout; subsidy does not cover cost."),
            SegmentSummary("Sleeping Dogs", 0, 0.0, 0.0, 0.0, "Do not discount; offer triggers negative response."),
        ]

    # Causal uplift quadrant classification
    # 1. Sleeping Dogs: ITE < 0
    sleeping_dogs_mask = ites < 0.0

    # 2. Persuadables: ITE > 0 and (ITE * value - cost) > 0
    persuadables_mask = (ites > 0.0) & ((ites * values - costs) > 0.0)

    # 3. Sure Things: Not in Persuadables/Sleeping Dogs and baseline_prob >= 0.50
    sure_things_mask = (~sleeping_dogs_mask) & (~persuadables_mask) & (probs >= 0.50)

    # 4. Lost Causes: Remaining low-uplift, low-probability users
    lost_causes_mask = (~sleeping_dogs_mask) & (~persuadables_mask) & (~sure_things_mask)

    segment_definitions = [
        ("Persuadables", persuadables_mask, "Primary target for promotional spend."),
        ("Sure Things", sure_things_mask, "High baseline conversion; discount unnecessary."),
        ("Lost Causes", lost_causes_mask, "Holdout; subsidy does not cover cost."),
        ("Sleeping Dogs", sleeping_dogs_mask, "Do not discount; offer triggers negative response."),
    ]

    segments: List[SegmentSummary] = []
    for name, mask, action in segment_definitions:
        count = int(np.sum(mask))
        pop_pct = round((count / n_customers) * 100.0, 2)
        if count > 0:
            mean_ite = round(float(np.mean(ites[mask])), 4)
            mean_prob = round(float(np.mean(probs[mask])), 4)
        else:
            mean_ite = 0.0
            mean_prob = 0.0

        segments.append(
            SegmentSummary(
                segment_name=name,
                customer_count=count,
                population_pct=pop_pct,
                mean_ite=mean_ite,
                mean_baseline_prob=mean_prob,
                recommended_action=action,
            )
        )

    return segments


def generate_executive_report(request: OptimizationRequest) -> ExecutiveCampaignReport:
    """Generate an executive campaign report comparing SciPy optimization to blanket baseline.

    Parameters:
        request: OptimizationRequest containing budget, objective, and customer data points.

    Returns:
        ExecutiveCampaignReport detailing high-level financial metrics, uplift quadrants,
        headline summary, and bullet insights.
    """
    ite_vector = np.array([c.predicted_ite for c in request.customers], dtype=np.float64)
    cost_vector = np.array([c.treatment_cost for c in request.customers], dtype=np.float64)
    value_vector = np.array([c.customer_value for c in request.customers], dtype=np.float64)
    prob_vector = np.array([c.baseline_conversion_prob for c in request.customers], dtype=np.float64)
    customer_ids = np.array([c.customer_id for c in request.customers])

    # 1. Run SciPyConstrainedSolver
    scipy_solver = SciPyConstrainedSolver(
        budget=request.budget,
        objective=request.objective,
        min_spend_per_user=request.min_spend_per_user,
    )
    scipy_res = scipy_solver.solve(
        treatment_effects=ite_vector,
        treatment_costs=cost_vector,
        customer_ids=customer_ids,
        customer_values=value_vector,
    )

    # 2. Run BlanketDiscountSolver
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

    total_customers = len(request.customers)
    budget = round(float(request.budget), 2)
    total_spend = round(float(scipy_res.total_spend), 2)
    expected_incremental_revenue = round(float(scipy_res.expected_incremental_revenue), 2)

    scipy_net_profit = float(
        scipy_res.metadata.get(
            "expected_net_profit",
            scipy_res.expected_incremental_revenue - scipy_res.total_spend,
        )
    )
    blanket_net_profit = float(
        blanket_res.metadata.get(
            "expected_net_profit",
            blanket_res.expected_incremental_revenue - blanket_res.total_spend,
        )
    )

    expected_net_profit = round(scipy_net_profit, 2)
    savings_vs_blanket = round(scipy_net_profit - blanket_net_profit, 2)
    roi_pct = round((expected_net_profit / total_spend) * 100.0, 2) if total_spend > 0.0 else 0.0

    # 3. Classify customer segments
    segments = classify_customer_segments(
        baseline_probs=prob_vector,
        treatment_effects=ite_vector,
        treatment_costs=cost_vector,
        customer_values=value_vector,
    )

    # 4. Generate Headline Summary
    headline_summary = (
        f"Causal optimization generates ${expected_net_profit:,.2f} in net profit and saves "
        f"${savings_vs_blanket:,.2f} compared to blanket targeting."
    )

    # 5. Generate Actionable Bullet Insights
    seg_map = {s.segment_name: s for s in segments}
    persuadables = seg_map.get("Persuadables")
    sleeping_dogs = seg_map.get("Sleeping Dogs")
    sure_things = seg_map.get("Sure Things")
    lost_causes = seg_map.get("Lost Causes")

    bullet_insights: List[str] = []

    # Insight 1: Persuadables targeting & ROI
    if persuadables and persuadables.customer_count > 0:
        bullet_insights.append(
            f"Targeting {persuadables.customer_count} Persuadables ({persuadables.population_pct:.1f}% of cohort) "
            f"delivers ${expected_incremental_revenue:,.2f} incremental revenue with an estimated net ROI of {roi_pct:.1f}%."
        )
    else:
        bullet_insights.append(
            "No viable Persuadables detected in cohort; zero promotional allocation recommended."
        )

    # Insight 2: Spend efficiency & budget utilization
    spend_pct = round((total_spend / budget) * 100.0, 1) if budget > 0 else 0.0
    bullet_insights.append(
        f"Allocated ${total_spend:,.2f} of ${budget:,.2f} campaign budget ({spend_pct:.1f}% utilization) "
        f"prioritizing high-margin causal uplift."
    )

    # Insight 3: Sleeping dogs exclusion & churn avoidance
    if sleeping_dogs and sleeping_dogs.customer_count > 0:
        bullet_insights.append(
            f"Withheld discounts from {sleeping_dogs.customer_count} Sleeping Dogs ({sleeping_dogs.population_pct:.1f}% of cohort), "
            f"preventing negative customer churn and wasted promotional spend."
        )
    else:
        bullet_insights.append(
            "Zero Sleeping Dogs detected in cohort; no negative reaction risks observed."
        )

    # Insight 4: Sure things margin protection
    if sure_things and sure_things.customer_count > 0:
        bullet_insights.append(
            f"Protected margin by withholding subsidies from {sure_things.customer_count} Sure Things ({sure_things.population_pct:.1f}% of cohort) "
            f"who convert organically at >=50% baseline probability."
        )
    else:
        bullet_insights.append(
            f"Identified {lost_causes.customer_count if lost_causes else 0} Lost Causes to place on holdout, avoiding unprofitable subsidies."
        )

    return ExecutiveCampaignReport(
        total_customers=total_customers,
        budget=budget,
        total_spend=total_spend,
        expected_incremental_revenue=expected_incremental_revenue,
        expected_net_profit=expected_net_profit,
        roi_pct=roi_pct,
        savings_vs_blanket=savings_vs_blanket,
        segments=segments,
        headline_summary=headline_summary,
        bullet_insights=bullet_insights,
    )

