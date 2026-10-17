from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "candidate_product_month_panel.csv"


def validate_panel(df: pd.DataFrame) -> None:
    """Validate product-month panel structure and core numeric values."""
    df = df.copy()
    df["Month"] = pd.PeriodIndex(df["Month"], freq="M")
    df = df.sort_values(["StockCode", "Month"]).copy()

    print(f"Panel rows: {len(df):,}")

    duplicate_count = int(df.duplicated(["StockCode", "Month"]).sum())
    print(f"Duplicate product-month keys: {duplicate_count}")

    if duplicate_count:
        raise ValueError(
            f"Panel contains {duplicate_count} duplicate product-month keys."
        )

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

        if missing:
            raise ValueError(
                f"Product {code} is missing calendar months: {missing}"
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
        missing_count = int(df[column].isna().sum())
        nonpositive_count = int(df[column].le(0).sum())

        print(
            f"{column}: missing={missing_count}, "
            f"nonpositive={nonpositive_count}"
        )

        if missing_count or nonpositive_count:
            raise ValueError(
                f"Invalid values found in column {column!r}."
            )


def main() -> None:
    """Load the saved panel and validate it."""
    df = pd.read_csv(INPUT)
    validate_panel(df)


if __name__ == "__main__":
    main()