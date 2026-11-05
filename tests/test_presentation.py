import os
import unittest
from pathlib import Path


class TestPresentationDeckOutline(unittest.TestCase):
    """Verifies that docs/presentation_deck_outline.md is complete, valid, and meets evaluation standards."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.deck_path = cls.repo_root / "docs" / "presentation_deck_outline.md"

    def test_deck_outline_exists(self):
        """Test that docs/presentation_deck_outline.md exists and is a regular file."""
        self.assertTrue(
            self.deck_path.is_file(),
            f"Expected presentation deck outline at {self.deck_path} was not found."
        )

    def test_deck_minimum_line_count(self):
        """Test that the presentation deck has at least 150 lines."""
        with open(self.deck_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreaterEqual(
            len(lines),
            150,
            f"Expected presentation deck to contain >= 150 lines, found {len(lines)}."
        )

    def test_deck_all_ten_slide_headings(self):
        """Test that all 10 slide headings ('Slide 1:' through 'Slide 10:') exist."""
        with open(self.deck_path, "r", encoding="utf-8") as f:
            content = f.read()

        for slide_num in range(1, 11):
            expected_heading = f"Slide {slide_num}:"
            self.assertIn(
                expected_heading,
                content,
                f"Missing required slide heading: '{expected_heading}'."
            )

    def test_deck_includes_speaker_notes_for_each_slide(self):
        """Test that each of the 10 slides contains dedicated Speaker Notes."""
        with open(self.deck_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that there are at least 10 occurrences of 'Speaker Notes'
        speaker_notes_count = content.count("Speaker Notes")
        self.assertGreaterEqual(
            speaker_notes_count,
            10,
            f"Expected at least 10 'Speaker Notes' sections, found {speaker_notes_count}."
        )

        # Check that each slide block has its own speaker notes
        slides = content.split("## Slide ")
        # The first split element is the pre-slide header
        self.assertGreaterEqual(len(slides), 11, "Expected 10 slide blocks parsed by '## Slide '")
        for i in range(1, 11):
            slide_block = slides[i]
            self.assertIn(
                "Speaker Notes",
                slide_block,
                f"Slide {i} is missing a 'Speaker Notes' section."
            )

    def test_deck_includes_visual_blueprint_for_each_slide(self):
        """Test that each of the 10 slides contains a Visual Blueprint section."""
        with open(self.deck_path, "r", encoding="utf-8") as f:
            content = f.read()

        visual_blueprint_count = content.count("Visual Blueprint")
        self.assertGreaterEqual(
            visual_blueprint_count,
            10,
            f"Expected at least 10 'Visual Blueprint' sections, found {visual_blueprint_count}."
        )

    def test_deck_author_and_credentials(self):
        """Test that author name, role, and organization credentials are present."""
        with open(self.deck_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Aniket Kapgate", content)
        self.assertIn("M3/M4", content)
        self.assertIn("Group 31", content)
        self.assertIn("HiGHS", content)


if __name__ == "__main__":
    unittest.main()

