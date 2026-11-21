
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

    df["Month"] = pd.PeriodIndex(df["Month"], freq="M")
    df = df.sort_values(["StockCode", "Month"]).copy()

    grouped = df.groupby("StockCode", observed=True)

    df["previous_price"] = grouped["median_price"].shift(1)
    df["previous_month"] = grouped["Month"].shift(1)

    # Retain only consecutive calendar months.
    consecutive = (
        df["Month"].astype("int64")
        - df["previous_month"].astype("int64")
    ) == 1

    changes = df[
        consecutive
        & df["previous_price"].notna()
        & df["median_price"].ne(df["previous_price"])
    ].copy()

    changes["absolute_change"] = (
        changes["median_price"] - changes["previous_price"]
    ).abs()

    print(f"Product-month records: {len(df):,}")
    print(f"Products: {df['StockCode'].nunique():,}")
    print(
        "Consecutive-month median-price changes: "
        f"{len(changes):,}"
    )

    print("\n=== Changes by month ===")
    print(
        changes.groupby("Month")
        .agg(
            products_with_price_changes=("StockCode", "nunique"),
            total_change_records=("StockCode", "size"),
        )
        .sort_values(
            "products_with_price_changes",
            ascending=False
        )
        .head(15)
        .to_string()
    )

    print("\n=== Largest changes with at least 10 rows ===")

    robust = changes[
        (changes["transaction_rows"] >= 10)
    ].copy()

    print(
        robust.sort_values(
            "absolute_change",
            ascending=False
        )[
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
