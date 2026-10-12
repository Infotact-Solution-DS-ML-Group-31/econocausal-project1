
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "retail_sales.csv"

CANDIDATES = ["85099B", "47566", "20685", "48138"]


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "StockCode",
        "Description",
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
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype("string")

    df = df[
        df["StockCode"].isin(CANDIDATES)
        & df["InvoiceDate"].notna()
        & df["Price"].gt(0)
        & df["Quantity"].gt(0)
    ].copy()

    for code in CANDIDATES:
        product = df[df["StockCode"] == code].copy()
        print(f"\n{'=' * 60}\nPRODUCT: {code}")

        descriptions = (
            product["Description"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
        )
        print("\nTop descriptions:")
        print(descriptions.head(10).to_string())
        print(f"Distinct non-missing descriptions: {len(descriptions)}")

        monthly = (
            product.groupby("Month", observed=True)
            .agg(
                rows=("Price", "size"),
                median_price=("Price", "median"),
                min_price=("Price", "min"),
                max_price=("Price", "max"),
                distinct_prices=("Price", "nunique"),
            )
            .reset_index()
            .sort_values("Month")
        )

        monthly["previous_median"] = monthly["median_price"].shift(1)
        changes = monthly[
            monthly["previous_median"].notna()
            & monthly["median_price"].ne(monthly["previous_median"])
        ]

        print("\nMonths with changed median price:")
        if changes.empty:
            print("No changes.")
        else:
            print(
                changes[
                    [
                        "Month",
                        "rows",
                        "previous_median",
                        "median_price",
                        "min_price",
                        "max_price",
                        "distinct_prices",
                    ]
                ].to_string(index=False)
            )

        print("\nMonthly price distribution:")
        print(
            monthly[
                ["Month", "rows", "median_price", "min_price", "max_price"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
