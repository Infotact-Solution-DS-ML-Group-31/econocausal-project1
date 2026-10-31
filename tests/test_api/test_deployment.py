"""Deployment verification unit tests for containerization artifacts (Dockerfile and .dockerignore)."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestDeploymentArtifacts(unittest.TestCase):
    """Test suite verifying Dockerfile and .dockerignore configurations for production readiness."""

    def setUp(self):
        self.dockerfile_path = PROJECT_ROOT / "Dockerfile"
        self.dockerignore_path = PROJECT_ROOT / ".dockerignore"

    def test_dockerfile_exists_and_contains_required_instructions(self):
        """Verify Dockerfile exists and contains required instructions: FROM, COPY, EXPOSE, HEALTHCHECK, CMD."""
        self.assertTrue(self.dockerfile_path.exists(), "Dockerfile must exist at repository root.")

        content = self.dockerfile_path.read_text(encoding="utf-8")

        # Required Docker instructions
        required_instructions = ["FROM", "COPY", "EXPOSE", "HEALTHCHECK", "CMD"]
        for instruction in required_instructions:
            self.assertIn(
                instruction,
                content,
                f"Dockerfile must contain instruction '{instruction}'.",
            )

        # Base image verification
        self.assertIn("python:3.11-slim", content, "Dockerfile should use python:3.11-slim base image.")

        # Multi-stage build check (at least 2 FROM instructions)
        from_count = len([line for line in content.splitlines() if line.strip().startswith("FROM")])
        self.assertGreaterEqual(from_count, 2, "Dockerfile must use a multi-stage build pattern.")

        # Port and networking verification
        self.assertIn("EXPOSE 8000", content, "Dockerfile must expose port 8000.")

        # Environment variables
        self.assertIn("PYTHONUNBUFFERED=1", content, "Dockerfile must set PYTHONUNBUFFERED=1.")
        self.assertIn("PORT=8000", content, "Dockerfile must configure PORT=8000.")

        # Healthcheck probe verification
        self.assertIn("HEALTHCHECK", content, "Dockerfile must declare HEALTHCHECK probe.")
        self.assertIn("http://localhost:8000/health", content, "HEALTHCHECK must target /health endpoint.")
        self.assertIn("curl", content, "HEALTHCHECK should execute curl probe.")

        # Server launch command verification
        self.assertIn("uvicorn", content, "CMD must launch Uvicorn ASGI server.")
        self.assertIn("src.api.main:app", content, "CMD must point to src.api.main:app.")

    def test_dockerignore_exists_and_excludes_required_patterns(self):
        """Verify .dockerignore exists and ignores .git, .venv, tests, caches, and scratch files."""
        self.assertTrue(self.dockerignore_path.exists(), ".dockerignore must exist at repository root.")

        content = self.dockerignore_path.read_text(encoding="utf-8")
        patterns = {line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")}

        # Critical exclusion checks
        self.assertIn(".git", patterns, ".dockerignore must exclude .git directory.")
        self.assertIn(".venv", patterns, ".dockerignore must exclude .venv directory.")

        # Additional production hygiene exclusions
        self.assertIn("__pycache__", patterns, ".dockerignore must exclude __pycache__.")
        self.assertIn(".pytest_cache", patterns, ".dockerignore must exclude .pytest_cache.")
        self.assertIn(".coverage", patterns, ".dockerignore must exclude .coverage.")
        self.assertIn("tests/", patterns, ".dockerignore must exclude tests/ directory.")


if __name__ == "__main__":
    unittest.main()

