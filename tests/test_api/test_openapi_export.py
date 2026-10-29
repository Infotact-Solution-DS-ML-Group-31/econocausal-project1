"""Unit tests for automated OpenAPI schema exporter and API Markdown summary generator."""

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

    def test_export_openapi_json_creates_valid_schema(self):
        """Verify export_openapi_json creates valid OpenAPI 3.x JSON with expected paths and schemas."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "schemas" / "openapi.json"
            result_path = export_openapi_json(output_path=str(out_file))

            self.assertEqual(result_path, str(out_file))
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Verify OpenAPI version
            self.assertIn("openapi", data)
            self.assertTrue(data["openapi"].startswith("3."))

            # Verify Paths
            self.assertIn("paths", data)
            self.assertIn("/api/v1/allocate", data["paths"])
            self.assertIn("/api/v1/allocate-multi-tier", data["paths"])
            self.assertIn("/api/v1/compare-policies", data["paths"])
            self.assertIn("/api/v1/drift-check", data["paths"])
            self.assertIn("/api/v1/simulate-confidence", data["paths"])
            self.assertIn("/api/v1/executive-report", data["paths"])
            self.assertIn("/api/v1/uplift-curve", data["paths"])
            self.assertIn("/health", data["paths"])

            # Verify Component Schemas
            self.assertIn("components", data)
            self.assertIn("schemas", data["components"])
            schemas = data["components"]["schemas"]

            self.assertIn("OptimizationRequest", schemas)
            self.assertIn("OptimizationResponse", schemas)
            self.assertIn("MultiTierOptimizationRequest", schemas)
            self.assertIn("MonteCarloSimulationRequest", schemas)
            self.assertIn("MonteCarloSimulationResponse", schemas)
            self.assertIn("ExecutiveReportResponse", schemas)
            self.assertIn("SegmentSummarySchema", schemas)

    def test_generate_api_markdown_summary_lists_primary_endpoints(self):
        """Verify generate_api_markdown_summary generates table listing all primary endpoints and request models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "docs" / "api_reference.md"
            result_path = generate_api_markdown_summary(output_path=str(out_file))

            self.assertEqual(result_path, str(out_file))
            self.assertTrue(out_file.exists())

            content = out_file.read_text(encoding="utf-8")

            # Verify Table Header
            self.assertIn("# EconoCausal API", content)
            self.assertIn("## Summary of Endpoints", content)
            self.assertIn("| HTTP Method | Endpoint | Request Model | Description |", content)

            # Verify Primary Endpoints
            expected_endpoints = [
                "/api/v1/allocate",
                "/api/v1/allocate-multi-tier",
                "/api/v1/compare-policies",
                "/api/v1/drift-check",
                "/api/v1/simulate-confidence",
                "/api/v1/executive-report",
                "/api/v1/uplift-curve",
                "/health",
                "/ready",
            ]
            for ep in expected_endpoints:
                self.assertIn(f"`{ep}`", content, f"Missing endpoint `{ep}` in Markdown table")

            # Verify Request Models
            self.assertIn("`OptimizationRequest`", content)
            self.assertIn("`MultiTierOptimizationRequest`", content)
            self.assertIn("`MonteCarloSimulationRequest`", content)
            self.assertIn("`DriftCheckRequest`", content)

    def test_default_export_execution(self):
        """Verify calling export functions with default paths creates docs files successfully."""
        json_path = export_openapi_json()
        self.assertTrue(Path(json_path).exists())

        md_path = generate_api_markdown_summary()
        self.assertTrue(Path(md_path).exists())


if __name__ == "__main__":
    unittest.main()

