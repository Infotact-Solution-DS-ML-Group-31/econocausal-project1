
"""Measure observed price variation within products."""

from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("data/processed/retail_sales.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run the dataset builder first."
        )

    data = pd.read_csv(path)
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )

    data = data.dropna(subset=["StockCode", "Price", "InvoiceDate"])
    data = data.loc[data["Price"] > 0].copy()

    # A product with multiple observed prices has variation, but this
    # alone does not establish that the variation is exogenous.
    products = data.groupby("StockCode").agg(
        rows=("Price", "size"),
        distinct_prices=("Price", "nunique"),
        min_price=("Price", "min"),
        max_price=("Price", "max"),
        median_price=("Price", "median"),
        distinct_months=(
            "InvoiceDate",
            lambda dates: dates.dt.to_period("M").nunique(),
        ),
    )

    products["absolute_price_range"] = (
        products["max_price"] - products["min_price"]
    )
    products["relative_price_range"] = (
        products["absolute_price_range"]
        / products["median_price"].replace(0, float("nan"))
    )

    variable_price = products["distinct_prices"] > 1

    print(f"Transaction rows analyzed: {len(data):,}")
    print(f"Unique products: {len(products):,}")
    print(f"Products with multiple observed prices: {variable_price.sum():,}")
    print(
        "Share with multiple prices: "
        f"{variable_price.mean() * 100:.2f}%"
    )
    print(
        "Products observed in at least 6 distinct months: "
        f"{(products['distinct_months'] >= 6).sum():,}"
    )

    print("\n=== Products with the most distinct prices ===")
    print(
        products.sort_values(
            ["distinct_prices", "rows"], ascending=False
        ).head(15).to_string()
    )

    print("\n=== Price variation by observation count ===")
    print(
        products.groupby(
            pd.cut(
                products["rows"],
                bins=[0, 5, 20, 100, float("inf")],
                labels=["1–5", "6–20", "21–100", "100+"],
            ),
            observed=False,
        ).agg(
            products=("rows", "size"),
            products_with_multiple_prices=(
                "distinct_prices",
                lambda values: (values > 1).sum(),
            ),
        ).to_string()
    )

    output = Path("data/processed/product_price_variation.csv")
    products.sort_values("rows", ascending=False).to_csv(output)
    print(f"\nSaved product summary to: {output}")


if __name__ == "__main__":
    main()
