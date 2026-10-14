"""Tests for FastAPI core runner and health/readiness endpoints."""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app


class TestHealthEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("docs", data)

    def test_health_liveness_probe(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)

    def test_ready_readiness_probe(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ready"])
        self.assertEqual(data["subsystems"]["optimization_engine"], "initialized")

    def test_process_time_header(self):
        response = self.client.get("/health")
        self.assertIn("X-Process-Time-Sec", response.headers)


if __name__ == "__main__":
    unittest.main()