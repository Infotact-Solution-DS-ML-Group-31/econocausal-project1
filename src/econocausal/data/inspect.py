
"""Inspect potentially problematic retail transaction records."""

from pathlib import Path

import pandas as pd

from econocausal.data.eda import load_retail_workbook


def main() -> None:
    path = Path("data/raw/online_retail_II.xlsx")
    data = load_retail_workbook(path)

    # Convert relevant fields to predictable types.
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    invoice = data["Invoice"].astype("string").str.strip()
    cancellation = invoice.str.startswith("C", na=False)

    checks = {
        "exact_duplicate_rows": data.duplicated().sum(),
        "cancellation_marked_rows": cancellation.sum(),
        "negative_quantity_rows": data["Quantity"].lt(0).sum(),
        "zero_quantity_rows": data["Quantity"].eq(0).sum(),
        "negative_price_rows": data["Price"].lt(0).sum(),
        "zero_price_rows": data["Price"].eq(0).sum(),
        "missing_customer_ids": data["Customer ID"].isna().sum(),
        "missing_descriptions": data["Description"].isna().sum(),
        "cancellation_and_negative_quantity": (
            cancellation & data["Quantity"].lt(0)
        ).sum(),
        "positive_quantity_and_nonpositive_price": (
            data["Quantity"].gt(0) & data["Price"].le(0)
        ).sum(),
        "negative_quantity_and_positive_price": (
            data["Quantity"].lt(0) & data["Price"].gt(0)
        ).sum(),
    }

    print("=== Diagnostic counts ===")
    for name, count in checks.items():
        print(f"{name}: {int(count):,}")

    print("\n=== Quantity quantiles ===")
    print(data["Quantity"].quantile(
        [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    ).to_string())

    print("\n=== Price quantiles ===")
    print(data["Price"].quantile(
        [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    ).to_string())

    print("\n=== Largest absolute quantities ===")
    print(
        data.loc[data["Quantity"].abs().nlargest(10).index,
                 ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]]
        .to_string(index=False)
    )

    print("\n=== Largest absolute prices ===")
    print(
        data.loc[data["Price"].abs().nlargest(10).index,
                 ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
