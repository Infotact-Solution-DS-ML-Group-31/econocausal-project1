
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT = ROOT / "data" / "processed" / "missing_customer_sensitivity.csv"

CANDIDATES = ["85099B", "20685", "48138", "47566"]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
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

    return df


def summarize(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return (
        df.groupby(["StockCode", "Month"], observed=True)
        .agg(
            transaction_rows=("Invoice", "size"),
            median_unit_price=("Price", "median"),
            mean_unit_price=("Price", "mean"),
            total_quantity_sold=("Quantity", "sum"),
            gross_line_value=("LineValue", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                column: f"{column}_{suffix}"
                for column in [
                    "transaction_rows",
                    "median_unit_price",
                    "mean_unit_price",
                    "total_quantity_sold",
                    "gross_line_value",
                ]
            }
        )
    )


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    df = prepare_data(df)

    all_transactions = summarize(df, "all")
    identified = summarize(
        df[df["Customer ID"].notna()],
        "identified",
    )

    comparison = all_transactions.merge(
        identified,
        on=["StockCode", "Month"],
        how="left",
    )

    comparison["median_price_difference"] = (
        comparison["median_unit_price_all"]
        - comparison["median_unit_price_identified"]
    )

    comparison["quantity_difference"] = (
        comparison["total_quantity_sold_all"]
        - comparison["total_quantity_sold_identified"]
    )

    comparison["gross_value_difference"] = (
        comparison["gross_line_value_all"]
        - comparison["gross_line_value_identified"]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT, index=False)

    print("=== Missing Customer ID sensitivity ===")
    print(f"Candidate transaction rows: {len(df):,}")
    print(f"Missing Customer IDs: {df['Customer ID'].isna().sum():,}")

    print(
        "Product-months with changed median price: "
        f"{comparison['median_price_difference'].ne(0).sum():,}"
    )

    print("\n=== Product-level comparison ===")
    print(
        comparison.groupby("StockCode").agg(
            product_months=("Month", "nunique"),
            changed_median_months=(
                "median_price_difference",
                lambda x: x.ne(0).sum(),
            ),
            quantity_difference=("quantity_difference", "sum"),
            gross_value_difference=("gross_value_difference", "sum"),
        ).to_string()
    )

    print(f"\nSaved report to: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
