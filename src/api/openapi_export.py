"""Automated OpenAPI schema exporter and API reference Markdown generator."""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from fastapi.routing import APIRoute

# Ensure project root is in sys.path when run directly as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


def export_openapi_json(output_path: str = "docs/openapi.json") -> str:
    """Generate and export the complete OpenAPI specification as formatted JSON.

    Parameters:
        output_path: Target file path to write the OpenAPI JSON specification.

    Returns:
        String path of the exported OpenAPI JSON file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    openapi_schema = app.openapi()

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    return str(dest)


def _extract_api_routes(routes: List[Any]) -> List[APIRoute]:
    """Recursively extract all APIRoute instances from FastAPI application routes."""
    extracted: List[APIRoute] = []
    for r in routes:
        if isinstance(r, APIRoute):
            extracted.append(r)
        elif hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
            extracted.extend(_extract_api_routes(r.original_router.routes))
        elif hasattr(r, "routes"):
            extracted.extend(_extract_api_routes(r.routes))
    return extracted


def generate_api_markdown_summary(output_path: str = "docs/api_reference.md") -> str:
    """Generate Markdown summary table of all registered API endpoints for frontend integration.

    Parameters:
        output_path: Target file path to write the API Markdown documentation.

    Returns:
        String path of the exported Markdown documentation.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    openapi = app.openapi()
    title = openapi.get("info", {}).get("title", "EconoCausal API Reference")
    version = openapi.get("info", {}).get("version", "0.1.0")
    description = openapi.get(
        "info", {}
    ).get("description", "Prescriptive AI Dynamic Pricing & Discount Allocation Engine")

    lines = [
        f"# {title}",
        "",
        f"**Version:** {version}  ",
        f"**Description:** {description}",
        "",
        "## Summary of Endpoints",
        "",
        "| HTTP Method | Endpoint Path | Tag / Category | Description |",
        "|:---|:---|:---|:---|",
    ]

    ignored_paths = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

    # Inspect all registered endpoints via app.routes
    api_routes = _extract_api_routes(app.routes)

    # Dedup and format rows while preserving route discovery order
    seen_endpoints = set()
    for route in api_routes:
        if route.path in ignored_paths:
            continue

        valid_methods = [m for m in route.methods if m not in ("HEAD", "OPTIONS")]
        for method in sorted(valid_methods):
            key = (method, route.path)
            if key in seen_endpoints:
                continue
            seen_endpoints.add(key)

            tag = ", ".join(route.tags) if route.tags else "General"

            summary = route.summary or route.description or ""
            if not summary and route.endpoint and route.endpoint.__doc__:
                summary = route.endpoint.__doc__.strip().split("\n")[0]
            if not summary:
                if route.path == "/":
                    summary = "Root service metadata and online status"
                elif route.path == "/health":
                    summary = "Liveness probe for health monitoring"
                elif route.path == "/ready":
                    summary = "Readiness probe verifying subsystem operational readiness"

            summary = summary.replace("|", "\\|").strip()
            lines.append(f"| `{method}` | `{route.path}` | {tag} | {summary} |")

    content = "\n".join(lines) + "\n"
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)

    return str(dest)


if __name__ == "__main__":
    json_path = export_openapi_json()
    print(f"Exported OpenAPI JSON to: {json_path}")
    md_path = generate_api_markdown_summary()
    print(f"Exported API Markdown summary to: {md_path}")
