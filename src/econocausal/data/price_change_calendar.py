
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "monthly_product_prices.csv"


def main():
    df = pd.read_csv(INPUT)

    required = {"StockCode", "Month", "median_price", "transaction_rows"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(["StockCode", "Month"]).copy()

    df["previous_price"] = (
        df.groupby("StockCode")["median_price"].shift(1)
    )
    df["price_changed"] = (
        df["previous_price"].notna()
        & df["median_price"].ne(df["previous_price"])
    )

    changes = df[df["price_changed"]].copy()

    print(f"Product-month records: {len(df):,}")
    print(f"Products: {df['StockCode'].nunique():,}")
    print(f"Observed month-to-month median-price changes: {len(changes):,}")

    print("\n=== Changes by month ===")
    print(
        changes.groupby("Month")
        .agg(
            products_with_price_changes=("StockCode", "nunique"),
            total_change_records=("StockCode", "size"),
        )
        .sort_values("products_with_price_changes", ascending=False)
        .head(15)
        .to_string()
    )

    print("\n=== Largest absolute median-price changes ===")
    changes["absolute_change"] = (
        changes["median_price"] - changes["previous_price"]
    ).abs()
    print(
        changes.sort_values("absolute_change", ascending=False)[
            [
                "StockCode",
                "Month",
                "previous_price",
                "median_price",
                "transaction_rows",
                "absolute_change",
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
