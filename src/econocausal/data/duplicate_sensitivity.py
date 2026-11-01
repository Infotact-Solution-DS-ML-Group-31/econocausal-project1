
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT = ROOT / "data" / "processed" / "duplicate_sensitivity_report.csv"

CANDIDATES = ["85099B", "20685", "48138", "47566"]


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["StockCode"] = df["StockCode"].astype("string").str.strip()

    df = df[
        df["StockCode"].isin(CANDIDATES)
        & df["InvoiceDate"].notna()
        & df["Price"].gt(0)
        & df["Quantity"].gt(0)
    ].copy()

    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["gross_line_value"] = df["Price"] * df["Quantity"]

    return (
        df.groupby(["StockCode", "Month"], observed=True)
        .agg(
            transaction_rows=("Price", "size"),
            median_unit_price=("Price", "median"),
            total_quantity_sold=("Quantity", "sum"),
            gross_line_value=("gross_line_value", "sum"),
        )
        .reset_index()
    )

def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)

    candidates = df[
        df["StockCode"].astype("string").str.strip().isin(CANDIDATES)
    ].copy()

    deduplicated = candidates.drop_duplicates(keep="first").copy()

    original_summary = build_monthly_summary(candidates)
    deduplicated_summary = build_monthly_summary(deduplicated)

    comparison = original_summary.merge(
        deduplicated_summary,
        on=["StockCode", "Month"],
        how="outer",
        suffixes=("_with_duplicates", "_without_duplicates"),
    )

    comparison["median_price_changed"] = (
        comparison["median_unit_price_with_duplicates"]
        != comparison["median_unit_price_without_duplicates"]
    )

    comparison["quantity_difference"] = (
        comparison["total_quantity_sold_with_duplicates"]
        - comparison["total_quantity_sold_without_duplicates"]
    )

    comparison["gross_value_difference"] = (
        comparison["gross_line_value_with_duplicates"]
        - comparison["gross_line_value_without_duplicates"]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT, index=False)

    print("=== Duplicate sensitivity summary ===")
    print(f"Candidate transaction rows: {len(candidates):,}")
    print(f"Exact duplicate rows: {len(candidates) - len(deduplicated):,}")
    print(
        "Product-month records with changed median price: "
        f"{comparison['median_price_changed'].sum():,}"
    )
    print(
        "Total quantity difference: "
        f"{comparison['quantity_difference'].sum():,.2f}"
    )
    print(
        "Gross line-value difference: "
        f"{comparison['gross_value_difference'].sum():,.2f}"
    )

    print("\n=== Product-level changes ===")
    print(
        comparison.groupby("StockCode").agg(
            product_months=("Month", "nunique"),
            changed_median_months=("median_price_changed", "sum"),
            quantity_difference=("quantity_difference", "sum"),
            gross_value_difference=("gross_value_difference", "sum"),
        ).to_string()
    )

    print(f"\nSaved report to: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
