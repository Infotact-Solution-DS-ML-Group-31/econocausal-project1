
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT = ROOT / "data" / "processed" / "monthly_product_prices.csv"

EXCLUDED_CODES = {
    "DOT",
    "POST",
    "ADJUST",
    "B",
    "M",
    "BANK CHARGES",
    "AMAZONFEE",
}


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    required = {"StockCode", "InvoiceDate", "Price", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["StockCode"] = df["StockCode"].astype("string").str.strip()

    df = df.dropna(
        subset=["StockCode", "InvoiceDate", "Price", "Quantity"]
    )
    df = df[(df["Price"] > 0) & (df["Quantity"] > 0)]
    df = df[~df["StockCode"].isin(EXCLUDED_CODES)].copy()

    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["StockCode", "Month"], observed=True)
        .agg(
            transaction_rows=("Price", "size"),
            median_price=("Price", "median"),
            distinct_prices=("Price", "nunique"),
            total_quantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    product_summary = (
        monthly.groupby("StockCode", observed=True)
        .agg(
            active_months=("Month", "nunique"),
            min_monthly_price=("median_price", "min"),
            max_monthly_price=("median_price", "max"),
            distinct_monthly_prices=("median_price", "nunique"),
            total_rows=("transaction_rows", "sum"),
        )
    )

    # Focus on frequently observed products across multiple months.
    eligible = product_summary[
        (product_summary["total_rows"] >= 100)
        & (product_summary["active_months"] >= 6)
        & (product_summary["distinct_monthly_prices"] > 1)
    ].sort_values(
        ["active_months", "total_rows"], ascending=False
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT, index=False)

    print(f"Rows after exclusions: {len(df):,}")
    print(f"Products after exclusions: {df['StockCode'].nunique():,}")
    print(f"Monthly product records: {len(monthly):,}")
    print(
        "Products with 100+ rows, 6+ active months, "
        f"and changing monthly median prices: {len(eligible):,}"
    )
    print("\n=== Top 15 candidate products ===")
    print(eligible.head(15).to_string())
    print(f"\nSaved monthly data to: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
