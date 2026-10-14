"""Tests for end-to-end payload serialization and numerical extraction."""

import json
import unittest
import numpy as np
from src.api.mock_data import generate_mock_customers, generate_mock_optimization_request
from src.api.schemas import (
    CustomerAllocation,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResponse,
)


class TestApiSerialization(unittest.TestCase):
    def test_mock_customer_generation(self):
        customers = generate_mock_customers(num_customers=50, seed=123)
        self.assertEqual(len(customers), 50)
        self.assertEqual(customers[0].customer_id, "cust_00001")

        # Check probability bounds
        for c in customers:
            self.assertGreaterEqual(c.baseline_conversion_prob, 0.0)
            self.assertLessEqual(c.baseline_conversion_prob, 1.0)
            self.assertGreater(c.treatment_cost, 0.0)

    def test_request_to_numpy_vector_extraction(self):
        """Verify M3 can reliably extract NumPy vectors for the SciPy optimizer."""
        request = generate_mock_optimization_request(num_customers=25, budget=150.0)

        # Extract vectors for solver
        customer_ids = np.array([c.customer_id for c in request.customers])
        ite_vector = np.array([c.predicted_ite for c in request.customers], dtype=np.float64)
        cost_vector = np.array([c.treatment_cost for c in request.customers], dtype=np.float64)
        value_vector = np.array([c.customer_value for c in request.customers], dtype=np.float64)

        self.assertEqual(ite_vector.shape, (25,))
        self.assertEqual(cost_vector.shape, (25,))
        self.assertEqual(len(customer_ids), 25)

        # Vectorized expected incremental profit calculation: (ITE * Value) - Cost
        expected_profit_delta = (ite_vector * value_vector) - cost_vector
        self.assertEqual(expected_profit_delta.shape, (25,))

    def test_full_roundtrip_json_serialization(self):
        """Verify JSON -> Pydantic -> Response -> JSON round-trip."""
        request = generate_mock_optimization_request(num_customers=10, budget=50.0)
        request_json = request.model_dump_json()

        # Re-parse from raw JSON string (simulating HTTP request payload)
        parsed_payload = json.loads(request_json)
        reconstructed_request = OptimizationRequest(**parsed_payload)
        self.assertEqual(len(reconstructed_request.customers), 10)
        self.assertEqual(reconstructed_request.budget, 50.0)

        # Construct dummy response
        allocations = [
            CustomerAllocation(
                customer_id=c.customer_id,
                assigned_discount=10.0 if i < 3 else 0.0,
                is_targeted=(i < 3),
                expected_incremental_revenue=c.predicted_ite * c.customer_value if i < 3 else 0.0,
                expected_net_profit=(c.predicted_ite * c.customer_value - 10.0) if i < 3 else 0.0,
            )
            for i, c in enumerate(reconstructed_request.customers)
        ]

        response = OptimizationResponse(
            success=True,
            status_message="Optimization completed",
            total_budget=50.0,
            total_spend=30.0,
            budget_utilization_pct=60.0,
            total_expected_incremental_revenue=sum(a.expected_incremental_revenue for a in allocations),
            total_expected_net_profit=sum(a.expected_net_profit for a in allocations),
            targeted_customers_count=3,
            total_customers_count=10,
            allocations=allocations,
            metadata={"solver": "MockGreedySolver"},
        )

        response_json = response.model_dump_json()
        self.assertIn("Optimization completed", response_json)
        self.assertIn("cust_00001", response_json)


if __name__ == "__main__":
    unittest.main()