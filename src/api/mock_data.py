"""Mock data generator for API contract verification and frontend scaffolding."""

from typing import List
import numpy as np
from src.api.schemas import CustomerDataPoint, OptimizationObjective, OptimizationRequest


def generate_mock_customers(
    num_customers: int = 100,
    seed: int = 42,
    treatment_cost: float = 10.0,
    customer_value: float = 50.0,
) -> List[CustomerDataPoint]:
    """Generate synthetic customer profiles with realistic ITE distributions.

    Models three customer behaviors:
      - Sleeping Dogs / Negative Responders (~10%): ITE < 0
      - Lost Causes / Invariant (~50%): ITE near 0
      - Persuadables (~40%): Positive uplift ITE in (0.05, 0.35)

    Args:
        num_customers: Total number of customer records to synthesize.
        seed: Random seed for reproducibility.
        treatment_cost: Cost of the promotional discount.
        customer_value: Average order value / conversion value.

    Returns:
        List of validated CustomerDataPoint instances.
    """
    rng = np.random.default_rng(seed)

    # Baseline conversion probabilities [0.02, 0.40]
    baseline_probs = rng.uniform(0.02, 0.40, size=num_customers)

    # Mixture distribution for treatment effects (ITE)
    # 10% negative uplift, 50% negligible uplift, 40% strong persuadables
    segments = rng.choice(["negative", "neutral", "persuadable"], size=num_customers, p=[0.10, 0.50, 0.40])

    ite_scores = np.zeros(num_customers, dtype=np.float64)
    for i, seg in enumerate(segments):
        if seg == "negative":
            ite_scores[i] = rng.uniform(-0.10, -0.01)
        elif seg == "neutral":
            ite_scores[i] = rng.uniform(-0.01, 0.03)
        else:  # persuadable
            ite_scores[i] = rng.uniform(0.05, 0.30)

    customers: List[CustomerDataPoint] = []
    for i in range(num_customers):
        customers.append(
            CustomerDataPoint(
                customer_id=f"cust_{i+1:05d}",
                baseline_conversion_prob=float(np.round(baseline_probs[i], 4)),
                predicted_ite=float(np.round(ite_scores[i], 4)),
                treatment_cost=treatment_cost,
                customer_value=customer_value,
            )
        )

    return customers


def generate_mock_optimization_request(
    num_customers: int = 100,
    budget: float = 300.0,
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_PROFIT,
) -> OptimizationRequest:
    """Convenience helper to generate a complete valid OptimizationRequest."""
    customers = generate_mock_customers(num_customers=num_customers)
    return OptimizationRequest(
        budget=budget,
        objective=objective,
        min_spend_per_user=0.0,
        customers=customers,
    )