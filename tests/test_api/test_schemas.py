"""Unit tests for Pydantic API data contracts."""

import unittest
from pydantic import ValidationError
from src.api.schemas import (
    CustomerDataPoint,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResponse,
    CustomerAllocation,
)


class TestApiSchemas(unittest.TestCase):
    def test_valid_customer_data_point(self):
        customer = CustomerDataPoint(
            customer_id="cust_001",
            baseline_conversion_prob=0.15,
            predicted_ite=0.08,
            treatment_cost=10.0,
            customer_value=60.0,
        )
        self.assertEqual(customer.customer_id, "cust_001")
        self.assertEqual(customer.predicted_ite, 0.08)

    def test_invalid_conversion_probability(self):
        with self.assertRaises(ValidationError):
            CustomerDataPoint(
                customer_id="cust_002",
                baseline_conversion_prob=1.5,  # Must be <= 1.0
                predicted_ite=0.05,
                treatment_cost=10.0,
            )

    def test_negative_treatment_cost(self):
        with self.assertRaises(ValidationError):
            CustomerDataPoint(
                customer_id="cust_003",
                baseline_conversion_prob=0.1,
                predicted_ite=0.05,
                treatment_cost=-5.0,  # Must be >= 0.0
            )

    def test_duplicate_customer_ids_rejected(self):
        customers = [
            CustomerDataPoint(
                customer_id="cust_dup",
                baseline_conversion_prob=0.2,
                predicted_ite=0.05,
                treatment_cost=10.0,
            ),
            CustomerDataPoint(
                customer_id="cust_dup",  # Duplicate ID
                baseline_conversion_prob=0.3,
                predicted_ite=0.10,
                treatment_cost=10.0,
            ),
        ]
        with self.assertRaises(ValidationError):
            OptimizationRequest(
                budget=500.0,
                objective=OptimizationObjective.MAXIMIZE_PROFIT,
                customers=customers,
            )

    def test_valid_optimization_request_and_response(self):
        request = OptimizationRequest(
            budget=100.0,
            objective=OptimizationObjective.MAXIMIZE_PROFIT,
            customers=[
                CustomerDataPoint(
                    customer_id="cust_101",
                    baseline_conversion_prob=0.1,
                    predicted_ite=0.12,
                    treatment_cost=10.0,
                )
            ],
        )
        self.assertEqual(request.budget, 100.0)

        # Verify Response structure
        response = OptimizationResponse(
            success=True,
            status_message="Optimal solution found",
            total_budget=100.0,
            total_spend=10.0,
            budget_utilization_pct=10.0,
            total_expected_incremental_revenue=6.0,
            total_expected_net_profit=-4.0,
            targeted_customers_count=1,
            total_customers_count=1,
            allocations=[
                CustomerAllocation(
                    customer_id="cust_101",
                    assigned_discount=10.0,
                    is_targeted=True,
                    expected_incremental_revenue=6.0,
                    expected_net_profit=-4.0,
                )
            ],
        )
        self.assertTrue(response.success)
        self.assertEqual(len(response.allocations), 1)


if __name__ == "__main__":
    unittest.main()