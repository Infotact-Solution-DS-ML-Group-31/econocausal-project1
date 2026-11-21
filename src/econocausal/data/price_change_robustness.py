
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "monthly_product_prices.csv"


def main():
    df = pd.read_csv(INPUT)

    df["Month"] = pd.PeriodIndex(df["Month"], freq="M")
    df = df.sort_values(["StockCode", "Month"]).copy()

    grouped = df.groupby("StockCode", observed=True)

    df["previous_price"] = grouped["median_price"].shift(1)
    df["previous_rows"] = grouped["transaction_rows"].shift(1)
    df["previous_month"] = grouped["Month"].shift(1)

    # Compare only consecutive calendar months.
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

    print(f"All observed consecutive-month price changes: {len(changes):,}")

    print("\n=== Robustness by minimum observations in BOTH months ===")
    for threshold in [1, 5, 10, 20, 30]:
        selected = changes[
            (changes["transaction_rows"] >= threshold)
            & (changes["previous_rows"] >= threshold)
        ]
        print(
            f"Minimum {threshold:>2} rows/month: "
            f"{len(selected):,} changes across "
            f"{selected['StockCode'].nunique():,} products"
        )

    robust = changes[
        (changes["transaction_rows"] >= 10)
        & (changes["previous_rows"] >= 10)
    ]

    print("\n=== Largest changes with at least 10 rows in both months ===")
    print(
        robust.sort_values("absolute_change", ascending=False)[
            [
                "StockCode",
                "Month",
                "previous_rows",
                "transaction_rows",
                "previous_price",
                "median_price",
                "absolute_change",
            ]
        ].head(20).to_string(index=False)
    )

    print("\n=== Candidate products under the same rule ===")
    candidates = robust[
        robust["StockCode"].astype(str).isin(
            ["85099B", "20685", "48138", "47566"]
        )
    ]
    print(
        candidates[
            [
                "StockCode",
                "Month",
                "previous_rows",
                "transaction_rows",
                "previous_price",
                "median_price",
                "absolute_change",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
