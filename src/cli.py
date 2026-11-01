"""Command-line interface for offline batch customer discount allocation."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.pipeline_bridge import (
    _COST_ALIASES,
    _CUSTOMER_ID_ALIASES,
    _extract_field,
    _ITE_ALIASES,
    _PROB_ALIASES,
    _VALUE_ALIASES,
    allocate_discounts,
    export_allocation_to_records,
)
from src.api.schemas import (
    CustomerDataPoint,
    OptimizationObjective,
    OptimizationRequest,
)


def load_customer_records(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load raw customer records from CSV or JSON file.

    Parameters:
        file_path: Path to input CSV or JSON file.

    Returns:
        List of customer record dictionaries.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "customers" in data and isinstance(data["customers"], list):
                return data["customers"]
            elif "records" in data and isinstance(data["records"], list):
                return data["records"]
            raise ValueError(
                "JSON input must be a list of records or contain a 'customers' or 'records' list."
            )
        else:
            raise ValueError("Unsupported JSON root structure (expected list or object).")

    # Default to CSV parsing
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            cleaned = {
                k.strip(): v.strip() if isinstance(v, str) else v
                for k, v in row.items()
                if k is not None
            }
            records.append(cleaned)
    return records


def parse_customer_datapoints(records: List[Dict[str, Any]]) -> List[CustomerDataPoint]:
    """Parse raw records into validated CustomerDataPoint instances.

    Parameters:
        records: List of dictionaries representing customer profiles.

    Returns:
        List of validated CustomerDataPoint instances.
    """
    if not records:
        raise ValueError("Input file contains no customer records.")

    customers: List[CustomerDataPoint] = []
    for idx, r in enumerate(records):
        cid_val = _extract_field(r, _CUSTOMER_ID_ALIASES)
        customer_id = str(cid_val) if cid_val is not None else f"cust_{idx + 1}"

        ite_raw = _extract_field(r, _ITE_ALIASES, default=0.0)
        try:
            ite = float(ite_raw)
        except (ValueError, TypeError):
            ite = 0.0

        prob_raw = _extract_field(r, _PROB_ALIASES, default=0.10)
        try:
            prob = float(prob_raw)
        except (ValueError, TypeError):
            prob = 0.10
        clamped_prob = max(0.0, min(1.0, prob))

        cost_raw = _extract_field(r, _COST_ALIASES, default=10.0)
        try:
            cost = float(cost_raw)
        except (ValueError, TypeError):
            cost = 10.0
        cost = max(0.0, cost)

        val_raw = _extract_field(r, _VALUE_ALIASES, default=50.0)
        try:
            val = float(val_raw)
        except (ValueError, TypeError):
            val = 50.0
        val = max(0.01, val)

        customers.append(
            CustomerDataPoint(
                customer_id=customer_id,
                baseline_conversion_prob=clamped_prob,
                predicted_ite=ite,
                treatment_cost=cost,
                customer_value=val,
            )
        )
    return customers


def write_allocations_output(
    records: List[Dict[str, Any]], output_path: Union[str, Path]
) -> str:
    """Write customer allocation results to CSV or JSON output file.

    Parameters:
        records: List of dictionary allocations.
        output_path: Target output path.

    Returns:
        Path of generated output file as string.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        fieldnames = [
            "customer_id",
            "assigned_discount",
            "is_targeted",
            "expected_incremental_revenue",
            "expected_net_profit",
        ]
        if records and "roi_pct" in records[0]:
            fieldnames.append("roi_pct")

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    return str(path)


def print_summary(
    total_customers: int,
    targeted_customers: int,
    budget: float,
    total_spend: float,
    expected_incremental_revenue: float,
    expected_net_profit: float,
    objective: str,
    output_path: str,
) -> None:
    """Print clean executive summary table to sys.stdout."""
    targeted_pct = (
        (targeted_customers / total_customers * 100.0) if total_customers > 0 else 0.0
    )
    budget_util_pct = (total_spend / budget * 100.0) if budget > 0 else 0.0
    roi_pct = (expected_net_profit / total_spend * 100.0) if total_spend > 0 else 0.0

    summary_lines = [
        "============================================================",
        "              EconoCausal Allocation Summary                ",
        "============================================================",
        f"Total Customers:          {total_customers:,}",
        f"Targeted Customers:       {targeted_customers:,} ({targeted_pct:.1f}%)",
        f"Campaign Budget:          ${budget:,.2f}",
        f"Total Spend:              ${total_spend:,.2f} ({budget_util_pct:.1f}% utilization)",
        f"Expected Inc. Revenue:    ${expected_incremental_revenue:,.2f}",
        f"Expected Net Profit:      ${expected_net_profit:,.2f} (ROI: {roi_pct:.1f}%)",
        f"Optimization Objective:   {objective}",
        f"Allocations Written To:   {output_path}",
        "============================================================",
    ]
    sys.stdout.write("\n".join(summary_lines) + "\n")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for EconoCausal batch allocation CLI."""
    parser = argparse.ArgumentParser(
        description="EconoCausal CLI: Offline batch customer discount allocation optimizer."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input CSV or JSON file containing customer records.",
    )
    parser.add_argument(
        "-b",
        "--budget",
        required=True,
        type=float,
        help="Campaign budget ceiling as a positive float.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="allocations_output.csv",
        help="Path to output CSV or JSON file where allocations will be written (default: allocations_output.csv).",
    )
    parser.add_argument(
        "--objective",
        default="maximize_profit",
        choices=["maximize_profit", "maximize_revenue", "maximize_conversions"],
        help="Target business objective (default: maximize_profit).",
    )
    parser.add_argument(
        "--min-spend",
        type=float,
        default=0.0,
        help="Minimum spend per targeted customer (default: 0.0).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint.

    Parameters:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code: 0 on success, non-zero on failure.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    # Validate budget
    if args.budget <= 0.0:
        sys.stderr.write(
            f"Error: Budget must be a positive float greater than 0.0 (received: {args.budget}).\n"
        )
        return 1

    # Validate min-spend
    if args.min_spend < 0.0:
        sys.stderr.write(
            f"Error: min-spend must be non-negative (received: {args.min_spend}).\n"
        )
        return 1

    # Load input records
    try:
        raw_records = load_customer_records(args.input)
    except Exception as e:
        sys.stderr.write(f"Error loading input file '{args.input}': {e}\n")
        return 1

    # Convert to CustomerDataPoints
    try:
        customers = parse_customer_datapoints(raw_records)
    except Exception as e:
        sys.stderr.write(f"Error parsing customer data: {e}\n")
        return 1

    # Build OptimizationRequest
    try:
        request = OptimizationRequest(
            budget=args.budget,
            objective=OptimizationObjective(args.objective),
            min_spend_per_user=args.min_spend,
            customers=customers,
        )
    except Exception as e:
        sys.stderr.write(f"Validation error in optimization request: {e}\n")
        return 1

    # Execute optimization via SciPy constrained solver
    try:
        response = allocate_discounts(request)
    except Exception as e:
        sys.stderr.write(f"Optimization engine error: {e}\n")
        return 1

    # Export allocation table
    flat_records = export_allocation_to_records(response)
    try:
        out_file = write_allocations_output(flat_records, args.output)
    except Exception as e:
        sys.stderr.write(f"Error writing output to '{args.output}': {e}\n")
        return 1

    # Print summary to stdout
    print_summary(
        total_customers=response.total_customers_count,
        targeted_customers=response.targeted_customers_count,
        budget=response.total_budget,
        total_spend=response.total_spend,
        expected_incremental_revenue=response.total_expected_incremental_revenue,
        expected_net_profit=response.total_expected_net_profit,
        objective=args.objective,
        output_path=out_file,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

