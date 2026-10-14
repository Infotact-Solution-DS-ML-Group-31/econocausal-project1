"""Unit tests for scoring and cohort summary endpoints."""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app


class TestScoringEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_mock_customers_default(self):
        response = self.client.get("/api/v1/mock-customers?count=25")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 25)
        self.assertEqual(data[0]["customer_id"], "cust_00001")

    def test_get_mock_customers_invalid_count(self):
        response = self.client.get("/api/v1/mock-customers?count=0")
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_score_summary_endpoint(self):
        # Fetch mock customers first
        customers = self.client.get("/api/v1/mock-customers?count=40").json()

        # Post to score-summary
        response = self.client.post("/api/v1/score-summary", json=customers)
        self.assertEqual(response.status_code, 200)
        summary = response.json()
        self.assertEqual(summary["total_customers"], 40)
        self.assertIn("profitable_persuadables_count", summary)
        self.assertIn("sleeping_dogs_count", summary)
        self.assertIn("recommended_max_budget", summary)


if __name__ == "__main__":
    unittest.main()