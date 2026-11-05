import os
import unittest
from pathlib import Path


class TestDocumentationVerification(unittest.TestCase):
    """Verifies that the final project report methodology dossier is complete and properly formatted."""

    @classmethod
    def setUpClass(cls):
        # Resolve repo root directory relative to this test file
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.report_path = cls.repo_root / "docs" / "final_report_methodology.md"

    def test_final_report_exists(self):
        """Test that docs/final_report_methodology.md exists and is readable."""
        self.assertTrue(
            self.report_path.is_file(),
            f"Expected methodology dossier at {self.report_path} was not found."
        )

    def test_final_report_minimum_line_count(self):
        """Test that the dossier has at least 200 lines of thorough content."""
        with open(self.report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreaterEqual(
            len(lines),
            200,
            f"Expected dossier to have at least 200 lines, found {len(lines)}."
        )

    def test_final_report_required_chapters(self):
        """Test that all 5 formal chapter headings are present in the dossier."""
        required_chapters = [
            "Chapter 1: Problem Definition & Business Context",
            "Chapter 2: Econometric & Mathematical Formulations",
            "Chapter 3: System Architecture & Full-Stack Implementation",
            "Chapter 4: Empirical Benchmark Results & Verification",
            "Chapter 5: Limitations, Future Work & Conclusion",
        ]
        with open(self.report_path, "r", encoding="utf-8") as f:
            content = f.read()

        for chapter in required_chapters:
            self.assertIn(
                chapter,
                content,
                f"Missing required chapter heading: '{chapter}'."
            )

    def test_final_report_cites_core_mathematical_concepts(self):
        """Test that core econometric and optimization concepts are mathematically cited."""
        required_tokens = [
            r"\tau_i",
            "HiGHS",
            "MCKP",
            "PSI",
        ]
        with open(self.report_path, "r", encoding="utf-8") as f:
            content = f.read()

        for token in required_tokens:
            self.assertIn(
                token,
                content,
                f"Missing required mathematical or algorithmic token: '{token}'."
            )

    def test_final_report_metadata(self):
        """Test that author and role metadata are explicitly declared."""
        with open(self.report_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Aniket Kapgate", content)
        self.assertIn("M3/M4", content)
        self.assertIn("Infotact Solutions", content)


if __name__ == "__main__":
    unittest.main()

