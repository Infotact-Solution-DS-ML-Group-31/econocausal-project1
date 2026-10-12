
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "retail_sales.csv"
OUTPUT = ROOT / "data" / "processed" / "candidate_quality_report.csv"

CANDIDATES = ["85099B", "20685", "48138", "47566"]


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    df["StockCode"] = df["StockCode"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype("string")

    df = df[
        df["StockCode"].isin(CANDIDATES)
        & df["InvoiceDate"].notna()
        & df["Price"].gt(0)
        & df["Quantity"].gt(0)
    ].copy()

    monthly = (
        df.groupby(["StockCode", "Month"], observed=True)
        .agg(
            rows=("Price", "size"),
            median_price=("Price", "median"),
            q1_price=("Price", lambda x: x.quantile(0.25)),
            q3_price=("Price", lambda x: x.quantile(0.75)),
            distinct_prices=("Price", "nunique"),
            distinct_descriptions=("Description", "nunique"),
        )
        .reset_index()
        .sort_values(["StockCode", "Month"])
    )

    monthly["previous_median"] = (
        monthly.groupby("StockCode")["median_price"].shift(1)
    )
    monthly["median_change"] = (
        monthly["median_price"] - monthly["previous_median"]
    )
    monthly["median_change_pct"] = (
        monthly["median_change"] / monthly["previous_median"] * 100
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT, index=False)
    monthly["price_iqr"] = monthly["q3_price"] - monthly["q1_price"]

    summary = (
        monthly.groupby("StockCode", observed=True)
        .agg(
            active_months=("Month", "nunique"),
            total_rows=("rows", "sum"),
            months_with_multiple_prices=(
                "distinct_prices",
                lambda x: int((x > 1).sum()),
            ),
            months_with_multiple_descriptions=(
                "distinct_descriptions",
                lambda x: int((x > 1).sum()),
            ),
            median_of_monthly_medians=("median_price", "median"),
            lowest_monthly_median=("median_price", "min"),
            highest_monthly_median=("median_price", "max"),
            largest_monthly_price_iqr=("price_iqr", "max"),
        )
    )

    print("=== Candidate summary ===")
    print(summary.to_string())
    print("\n=== Monthly price changes ===")
    print(
        monthly.loc[
            monthly["median_change"].notna()
            & monthly["median_change"].ne(0),
            [
                "StockCode",
                "Month",
                "rows",
                "median_price",
                "median_change",
                "median_change_pct",
                "distinct_prices",
                "distinct_descriptions",
            ],
        ].to_string(index=False)
    )
    print(f"\nSaved report to: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
