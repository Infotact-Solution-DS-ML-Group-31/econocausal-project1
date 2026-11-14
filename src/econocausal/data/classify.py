
"""Classify unusual Online Retail II records without modifying the data."""

from pathlib import Path

import pandas as pd

from econocausal.data.eda import load_retail_workbook


def main() -> None:
    data = load_retail_workbook(
        Path("data/raw/online_retail_II.xlsx")
    )

    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    invoice = data["Invoice"].astype("string").str.strip()
    stock = data["StockCode"].astype("string").str.strip().str.upper()

    data["is_cancellation"] = invoice.str.startswith("C", na=False)
    data["is_zero_price"] = data["Price"].eq(0)
    data["is_negative_price"] = data["Price"].lt(0)
    data["is_negative_quantity"] = data["Quantity"].lt(0)

    print("=== Rows by source sheet ===")
    print(data.groupby("SourceSheet").size().to_string())

    print("\n=== Most common stock codes ===")
    print(data["StockCode"].value_counts().head(20).to_string())

    print("\n=== Zero-price rows: common stock codes ===")
    print(
        data.loc[data["is_zero_price"], "StockCode"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\n=== Zero-price rows: common descriptions ===")
    print(
        data.loc[data["is_zero_price"], "Description"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    print("\n=== Exceptional stock codes ===")
    special_codes = {"B", "M", "BANK CHARGES", "AMAZONFEE"}
    special = data.loc[stock.isin(special_codes)]
    print(
        special.groupby("StockCode", dropna=False)
        .agg(
            rows=("Invoice", "size"),
            total_quantity=("Quantity", "sum"),
            min_price=("Price", "min"),
            max_price=("Price", "max"),
            cancellation_rows=("is_cancellation", "sum"),
        )
        .to_string()
    )

    print("\n=== Cancellation rows without negative quantity ===")
    print(
        data.loc[
            data["is_cancellation"] & ~data["is_negative_quantity"],
            ["Invoice", "StockCode", "Description", "Quantity", "Price"],
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n=== Exact duplicates: sample ===")
    duplicate_mask = data.duplicated(keep=False)
    print(
        data.loc[duplicate_mask]
        .sort_values(["Invoice", "StockCode", "InvoiceDate"])
        .head(20)
        .to_string(index=False)
    )

    
    print("\n=== Price and quantity summaries by transaction type ===")
    data["is_cancellation"] = (
        invoice.str.startswith("C", na=False).astype(bool)
    )

    summary = data.groupby(
        "is_cancellation",
        dropna=False,
    ).agg(
        rows=("Invoice", "size"),
        median_quantity=("Quantity", "median"),
        median_price=("Price", "median"),
        zero_price_rows=("is_zero_price", "sum"),
        negative_quantity_rows=("is_negative_quantity", "sum"),
    )

    print(summary.to_string()
          
    )



if __name__ == "__main__":
    main()
