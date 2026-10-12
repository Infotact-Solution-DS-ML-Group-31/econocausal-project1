
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "candidate_product_month_panel.csv"


def main():
    df = pd.read_csv(INPUT)

    df["Month"] = pd.PeriodIndex(df["Month"], freq="M")
    df = df.sort_values(["StockCode", "Month"]).copy()

    print(f"Panel rows: {len(df):,}")
    print(f"Duplicate product-month keys: {df.duplicated(['StockCode', 'Month']).sum()}")

    expected_months = pd.period_range(
        df["Month"].min(), df["Month"].max(), freq="M"
    )
    print(f"Expected calendar months: {len(expected_months)}")

    print("\n=== Calendar continuity ===")
    for code, product in df.groupby("StockCode", observed=True):
        months = set(product["Month"])
        missing = sorted(set(expected_months) - months)
        print(
            f"{code}: observed={len(months)}, "
            f"missing_months={len(missing)}, "
            f"missing={missing}"
        )

    df["previous_month"] = df.groupby("StockCode")["Month"].shift(1)
    df["previous_price"] = (
        df.groupby("StockCode")["median_unit_price"].shift(1)
    )

    consecutive = (
        df["Month"].astype("int64")
        - df["previous_month"].astype("int64")
    ) == 1

    valid_change = (
        consecutive
        & df["previous_price"].notna()
        & df["median_unit_price"].ne(df["previous_price"])
    )

    print("\n=== Consecutive-month price changes ===")
    print(
        df.assign(valid_change=valid_change)
        .groupby("StockCode", observed=True)["valid_change"]
        .sum()
        .to_string()
    )

    print("\n=== Basic numeric validation ===")
    for column in [
        "transaction_rows",
        "median_unit_price",
        "total_quantity_sold",
        "gross_line_value",
    ]:
        print(
            f"{column}: missing={df[column].isna().sum()}, "
            f"nonpositive={(df[column] <= 0).sum()}"
        )


if __name__ == "__main__":
    main()
