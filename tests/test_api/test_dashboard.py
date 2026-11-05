"""Unit tests for the EconoCausal interactive HTML5/Plotly dashboard endpoint."""

import unittest
from fastapi.testclient import TestClient

from src.api.main import app


class TestDashboardEndpoint(unittest.TestCase):
    """Test suite covering the GET /dashboard route and frontend integration."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_dashboard_returns_html_and_expected_components(self):
        """GET /dashboard must return HTTP 200, text/html, and core UI tokens."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)

        content_type = response.headers.get("content-type", "")
        self.assertIn("text/html", content_type)

        html_text = response.text
        # Assert required keyword tokens
        self.assertIn("EconoCausal", html_text)
        self.assertIn("Plotly", html_text)
        self.assertIn("Allocation Matrix", html_text)

        # Assert key interactive dashboard sections
        self.assertIn("Campaign Policy & Budget Controls", html_text)
        self.assertIn("Executive Insight", html_text)
        self.assertIn("Uplift & Qini Curve", html_text)
        self.assertIn("Budget Efficiency Frontier", html_text)
        self.assertIn("Persuadables", html_text)
        self.assertIn("Sleeping Dogs", html_text)

    def test_root_endpoint_references_dashboard(self):
        """Root GET / must include link to /dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dashboard", data)
        self.assertEqual(data["dashboard"], "/dashboard")


if __name__ == "__main__":
    unittest.main()

