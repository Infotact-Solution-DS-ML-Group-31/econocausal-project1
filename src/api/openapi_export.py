"""Automated OpenAPI schema exporter and API reference Markdown generator."""

import json
from pathlib import Path
import sys
from typing import Any, Dict

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
        "| HTTP Method | Endpoint | Request Model | Description |",
        "|:---|:---|:---|:---|",
    ]

    for path, methods in openapi.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue

            req_body = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            ref = req_body.get("$ref")
            if ref:
                model_name = ref.split("/")[-1]
            elif req_body.get("type") == "array":
                item_ref = req_body.get("items", {}).get("$ref", "")
                if item_ref:
                    model_name = f"List[{item_ref.split('/')[-1]}]"
                else:
                    model_name = "List[Any]"
            else:
                model_name = "None"

            summary = op.get("summary", "") or op.get("description", "")
            summary = summary.replace("|", "\\|").strip()

            lines.append(f"| `{method.upper()}` | `{path}` | `{model_name}` | {summary} |")

    content = "\n".join(lines) + "\n"
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)

    return str(dest)


if __name__ == "__main__":
    json_path = export_openapi_json()
    print(f"Exported OpenAPI JSON to: {json_path}")
    md_path = generate_api_markdown_summary()
    print(f"Exported API Markdown summary to: {md_path}")
