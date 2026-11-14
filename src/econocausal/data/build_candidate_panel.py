
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT = ROOT / "data" / "processed" / "candidate_product_month_panel.csv"

CANDIDATES = ["85099B", "20685", "48138", "47566"]


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "StockCode",
        "Description",
        "Invoice",
        "InvoiceDate",
        "Price",
        "Quantity",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    df = df[
        df["StockCode"].isin(CANDIDATES)
        & df["InvoiceDate"].notna()
        & df["Price"].gt(0)
        & df["Quantity"].gt(0)
    ].copy()

    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["LineValue"] = df["Price"] * df["Quantity"]

    panel = (
        df.groupby(["StockCode", "Month"], observed=True)
        .agg(
            transaction_rows=("Invoice", "size"),
            median_unit_price=("Price", "median"),
            mean_unit_price=("Price", "mean"),
            distinct_unit_prices=("Price", "nunique"),
            total_quantity_sold=("Quantity", "sum"),
            gross_line_value=("LineValue", "sum"),
            distinct_descriptions=("Description", "nunique"),
        )
        .reset_index()
        .sort_values(["StockCode", "Month"])
    )

    panel["previous_month_price"] = (
        panel.groupby("StockCode")["median_unit_price"].shift(1)
    )
    panel["price_change"] = (
        panel["median_unit_price"] - panel["previous_month_price"]
    )
    panel["price_change_pct"] = (
        panel["price_change"] / panel["previous_month_price"] * 100
    )

    # Flag, but do not automatically remove, low-support months.
    panel["at_least_10_rows"] = panel["transaction_rows"] >= 10
    panel["multiple_descriptions"] = panel["distinct_descriptions"] > 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT, index=False)

    print(f"Panel rows: {len(panel):,}")
    print(f"Products: {panel['StockCode'].nunique():,}")
    print(f"Month range: {panel['Month'].min()} to {panel['Month'].max()}")
    print("\n=== Product-level panel summary ===")
    print(
        panel.groupby("StockCode", observed=True)
        .agg(
            active_months=("Month", "nunique"),
            total_transactions=("transaction_rows", "sum"),
            months_with_10_plus_rows=("at_least_10_rows", "sum"),
            months_with_multiple_descriptions=("multiple_descriptions", "sum"),
            months_with_price_changes=("price_change", lambda x: x.ne(0).sum()),
        )
        .to_string()
    )
    print(f"\nSaved panel to: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
