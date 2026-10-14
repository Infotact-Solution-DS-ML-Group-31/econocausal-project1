"""API routes for customer ITE scoring and mock campaign previews."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, status
from src.api.mock_data import generate_mock_customers
from src.api.schemas import CustomerDataPoint

router = APIRouter(prefix="/api/v1", tags=["Scoring & Previews"])


@router.get(
    "/mock-customers",
    response_model=List[CustomerDataPoint],
    status_code=status.HTTP_200_OK,
    summary="Generate synthetic customer ITE dataset for UI testing",
)
async def get_mock_customers(
    count: int = Query(default=50, ge=1, le=5000, description="Number of customers to generate"),
    treatment_cost: float = Query(default=10.0, gt=0.0, description="Discount cost per customer"),
    customer_value: float = Query(default=50.0, gt=0.0, description="Expected order value"),
) -> List[CustomerDataPoint]:
    """Return synthetic customer records with Causal ITE scores for frontend integration."""
    return generate_mock_customers(
        num_customers=count,
        treatment_cost=treatment_cost,
        customer_value=customer_value,
    )


@router.post(
    "/score-summary",
    status_code=status.HTTP_200_OK,
    summary="Calculate aggregate uplift statistics across a customer cohort",
)
async def summarize_customer_scores(customers: List[CustomerDataPoint]) -> Dict[str, Any]:
    """Compute distributional metrics on incoming customer ITE scores before optimization."""
    if not customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer list cannot be empty.",
        )

    ites = [c.predicted_ite for c in customers]
    persuadables = [
        c for c in customers if (c.predicted_ite * c.customer_value) > c.treatment_cost
    ]
    sleeping_dogs = [c for c in customers if c.predicted_ite < 0]

    return {
        "total_customers": len(customers),
        "mean_ite": round(sum(ites) / len(ites), 4),
        "max_ite": round(max(ites), 4),
        "min_ite": round(min(ites), 4),
        "profitable_persuadables_count": len(persuadables),
        "sleeping_dogs_count": len(sleeping_dogs),
        "recommended_max_budget": round(sum(c.treatment_cost for c in persuadables), 2),
    }