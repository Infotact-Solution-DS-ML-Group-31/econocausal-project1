
"""Run the core EconoCausal data preparation workflow."""

from econocausal.data.build_dataset import main as build_sales_dataset
from econocausal.data.build_candidate_panel import main as build_candidate_panel
from econocausal.data.validate_candidate_panel import main as validate_candidate_panel


def main() -> None:
    """Build the prepared dataset, construct the panel, and validate it."""
    print("\n" + "=" * 60)
    print("STEP 1: BUILD PREPARED SALES DATASET")
    print("=" * 60)
    build_sales_dataset()

    print("\n" + "=" * 60)
    print("STEP 2: BUILD CANDIDATE PRODUCT-MONTH PANEL")
    print("=" * 60)
    build_candidate_panel()

    print("\n" + "=" * 60)
    print("STEP 3: VALIDATE CANDIDATE PANEL")
    print("=" * 60)
    validate_candidate_panel()

    print("\nPipeline stages completed.")


if __name__ == "__main__":
    main()
