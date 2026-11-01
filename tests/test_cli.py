"""Unit tests for offline batch discount allocation CLI."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import main


class TestCliBatchAllocation(unittest.TestCase):
    """Test suite verifying CLI argument parsing, file ingestion, solver execution, and output export."""

    def setUp(self):
        self.sample_customers = [
            {
                "customer_id": f"cust_{i+1:03d}",
                "baseline_conversion_prob": 0.15,
                "predicted_ite": 0.25 if i < 5 else (-0.10 if i < 8 else 0.02),
                "treatment_cost": 10.0,
                "customer_value": 60.0,
            }
            for i in range(10)
        ]

    def _write_csv_input(self, file_path: Path):
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "customer_id",
                    "baseline_conversion_prob",
                    "predicted_ite",
                    "treatment_cost",
                    "customer_value",
                ],
            )
            writer.writeheader()
            writer.writerows(self.sample_customers)

    def _write_json_input(self, file_path: Path, wrapper_key: str = None):
        data = (
            {wrapper_key: self.sample_customers}
            if wrapper_key
            else self.sample_customers
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def test_cli_csv_input_and_csv_output(self):
        """Verify CLI processes CSV input, enforces budget constraint, and writes valid CSV output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "input_customers.csv"
            out_file = Path(tmp_dir) / "output_allocations.csv"
            self._write_csv_input(in_file)

            exit_code = main(
                [
                    "--input",
                    str(in_file),
                    "--budget",
                    "30.0",
                    "--output",
                    str(out_file),
                    "--objective",
                    "maximize_profit",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            # Verify CSV content and columns
            with open(out_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 10)
            required_cols = [
                "customer_id",
                "assigned_discount",
                "is_targeted",
                "expected_incremental_revenue",
                "expected_net_profit",
            ]
            for col in required_cols:
                self.assertIn(col, reader.fieldnames)

            # Check budget compliance: total spend <= $30.0
            total_spend = sum(float(r["assigned_discount"]) for r in rows)
            self.assertLessEqual(total_spend, 30.0)
            self.assertGreater(total_spend, 0.0)

            # Targeted customers count: each cost is $10 -> at most 3 customers
            targeted_count = sum(1 for r in rows if r["is_targeted"].lower() in ("true", "1"))
            self.assertLessEqual(targeted_count, 3)

    def test_cli_json_input_and_json_output(self):
        """Verify CLI processes JSON input and writes valid JSON output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "input_customers.json"
            out_file = Path(tmp_dir) / "output_allocations.json"
            self._write_json_input(in_file)

            exit_code = main(
                [
                    "-i",
                    str(in_file),
                    "-b",
                    "40.0",
                    "-o",
                    str(out_file),
                    "--objective",
                    "maximize_revenue",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(len(data), 10)
            total_spend = sum(float(r["assigned_discount"]) for r in data)
            self.assertLessEqual(total_spend, 40.0)

    def test_cli_json_input_with_customers_wrapper(self):
        """Verify CLI supports JSON files wrapped in a 'customers' dictionary key."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "wrapped_customers.json"
            out_file = Path(tmp_dir) / "allocations.csv"
            self._write_json_input(in_file, wrapper_key="customers")

            exit_code = main(
                [
                    "-i",
                    str(in_file),
                    "-b",
                    "20.0",
                    "-o",
                    str(out_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

    def test_cli_invalid_negative_budget(self):
        """Verify CLI rejects negative budget with non-zero exit code."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "customers.csv"
            self._write_csv_input(in_file)

            exit_code = main(["-i", str(in_file), "-b", "-10.0"])
            self.assertNotEqual(exit_code, 0)

    def test_cli_invalid_zero_budget(self):
        """Verify CLI rejects zero budget with non-zero exit code."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "customers.csv"
            self._write_csv_input(in_file)

            exit_code = main(["-i", str(in_file), "-b", "0.0"])
            self.assertNotEqual(exit_code, 0)

    def test_cli_missing_input_file(self):
        """Verify CLI returns non-zero exit code when input file does not exist."""
        exit_code = main(["-i", "nonexistent_file_9999.csv", "-b", "50.0"])
        self.assertNotEqual(exit_code, 0)

    def test_cli_missing_required_arguments(self):
        """Verify CLI returns non-zero exit code when required arguments are omitted."""
        exit_code = main([])
        self.assertNotEqual(exit_code, 0)

    def test_cli_subprocess_execution(self):
        """Verify end-to-end execution of CLI script via python subprocess."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "sub_input.csv"
            out_file = Path(tmp_dir) / "sub_output.csv"
            self._write_csv_input(in_file)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "--input",
                    str(in_file),
                    "--budget",
                    "20.0",
                    "--output",
                    str(out_file),
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("EconoCausal Allocation Summary", result.stdout)
            self.assertIn("Total Customers:", result.stdout)
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()

