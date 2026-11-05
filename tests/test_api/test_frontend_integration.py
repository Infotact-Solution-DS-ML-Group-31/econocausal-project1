"""Integration test suite verifying Role M4 (Frontend) scaffolding and FastAPI integration."""

import json
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from src.api.main import app


class TestFrontendIntegration(unittest.TestCase):
    """Test suite covering M4 frontend assets and M3/M4 integration."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.frontend_dir = Path(__file__).resolve().parents[2] / "frontend"

    def test_frontend_scaffold_assets_exist(self):
        """Verify essential frontend scaffold assets exist in frontend/."""
        required_assets = [
            "package.json",
            "vite.config.js",
            "index.html",
            "standalone.html",
            "src/App.jsx",
            "src/main.jsx",
            "src/index.css",
            "src/api/client.js",
            "src/components/Navbar.jsx",
            "src/components/ControlPanel.jsx",
            "src/components/KPICards.jsx",
            "src/components/UpliftCharts.jsx",
            "src/components/AllocationMatrix.jsx",
            "src/components/ExecutiveInsight.jsx",
        ]

        for asset in required_assets:
            asset_path = self.frontend_dir / asset
            self.assertTrue(asset_path.is_file(), f"Required frontend asset missing: {asset}")

    def test_package_json_dependencies(self):
        """Verify package.json specifies required React, Plotly, Vite, and Tailwind dependencies."""
        pkg_file = self.frontend_dir / "package.json"
        self.assertTrue(pkg_file.is_file())

        with open(pkg_file, "r", encoding="utf-8") as f:
            pkg_data = json.load(f)

        deps = pkg_data.get("dependencies", {})
        dev_deps = pkg_data.get("devDependencies", {})

        self.assertIn("react", deps)
        self.assertIn("react-dom", deps)
        self.assertIn("plotly.js-dist-min", deps)
        self.assertIn("react-plotly.js", deps)
        self.assertIn("lucide-react", deps)

        self.assertIn("vite", dev_deps)
        self.assertIn("@vitejs/plugin-react", dev_deps)
        self.assertIn("tailwindcss", dev_deps)

    def test_get_dashboard_endpoint_returns_html_and_assets(self):
        """Test GET /dashboard returns HTTP 200 with text/html content-type and core UI tokens."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)

        content_type = response.headers.get("content-type", "")
        self.assertIn("text/html", content_type)

        content = response.text
        self.assertIn("EconoCausal", content)
        self.assertIn("Plotly", content)
        self.assertIn("Allocation Matrix", content)
        self.assertIn("Campaign Policy & Budget Controls", content)
        self.assertIn("Executive Insight", content)


if __name__ == "__main__":
    unittest.main()

