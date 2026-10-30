"""Unit tests for automated OpenAPI schema exporter and API Markdown reference generator."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.openapi_export import (
    export_openapi_json,
    generate_api_markdown_summary,
)


class TestOpenAPIExport(unittest.TestCase):
    """Test suite covering OpenAPI specification export and API reference documentation generation."""

    def test_export_openapi_json_contains_required_sections_and_paths(self):
        """Verify export_openapi_json creates valid JSON containing 'openapi', 'info', and 'paths'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "schemas" / "openapi.json"
            result_path = export_openapi_json(output_path=str(out_file))

            self.assertEqual(result_path, str(out_file))
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Assert root required OpenAPI keys
            self.assertIn("openapi", data)
            self.assertTrue(data["openapi"].startswith("3."))
            self.assertIn("info", data)
            self.assertIn("title", data["info"])
            self.assertIn("paths", data)

            # Assert key required endpoints in paths
            paths = data["paths"]
            self.assertIn("/api/v1/allocate", paths)
            self.assertIn("/api/v1/drift-check", paths)
            self.assertIn("/api/v1/allocate-multi-tier", paths)
            self.assertIn("/api/v1/compare-policies", paths)
            self.assertIn("/api/v1/executive-report", paths)
            self.assertIn("/api/v1/uplift-curve", paths)
            self.assertIn("/health", paths)
            self.assertIn("/ready", paths)

            # Assert component schemas are present
            self.assertIn("components", data)
            self.assertIn("schemas", data["components"])
            schemas = data["components"]["schemas"]
            self.assertIn("OptimizationRequest", schemas)
            self.assertIn("OptimizationResponse", schemas)

    def test_generate_api_markdown_summary_table_and_core_routes(self):
        """Verify generate_api_markdown_summary creates markdown with required table header and all 10 core routes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "docs" / "api_reference.md"
            result_path = generate_api_markdown_summary(output_path=str(out_file))

            self.assertEqual(result_path, str(out_file))
            self.assertTrue(out_file.exists())

            content = out_file.read_text(encoding="utf-8")

            # Assert table header contains '| HTTP Method | Endpoint Path |'
            self.assertIn("| HTTP Method | Endpoint Path |", content)
            self.assertIn("| HTTP Method | Endpoint Path | Tag / Category | Description |", content)

            # Assert documentation for all 10 core routes
            core_10_routes = [
                "/health",
                "/ready",
                "/api/v1/mock-customers",
                "/api/v1/score-summary",
                "/api/v1/allocate",
                "/api/v1/allocate-multi-tier",
                "/api/v1/compare-policies",
                "/api/v1/drift-check",
                "/api/v1/executive-report",
                "/api/v1/uplift-curve",
            ]
            for route_path in core_10_routes:
                self.assertIn(f"`{route_path}`", content, f"Core route {route_path} missing in markdown table")

            # Assert categories / tags are present
            self.assertIn("Monitoring", content)
            self.assertIn("Scoring & Previews", content)
            self.assertIn("Optimization & Allocation", content)

    def test_default_export_execution(self):
        """Verify calling export functions with default paths updates docs files on disk."""
        json_path = export_openapi_json()
        self.assertTrue(Path(json_path).exists())

        md_path = generate_api_markdown_summary()
        self.assertTrue(Path(md_path).exists())


if __name__ == "__main__":
    unittest.main()
