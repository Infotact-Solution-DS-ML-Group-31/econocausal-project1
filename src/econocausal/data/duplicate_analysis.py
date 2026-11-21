
"""Quantify the effect of exact duplicate rows on sales summaries."""

from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("data/processed/retail_sales.csv")
    data = pd.read_csv(path)

    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data["line_value"] = data["Quantity"] * data["Price"]

    duplicate_mask = data.duplicated(keep="first")
    deduplicated = data.loc[~duplicate_mask].copy()

    print(f"Rows before deduplication: {len(data):,}")
    print(f"Exact duplicate rows after first occurrence: {int(duplicate_mask.sum()):,}")
    print(f"Rows after removing exact duplicates: {len(deduplicated):,}")

    print("\n=== Quantity totals ===")
    print(f"With duplicates: {data['Quantity'].sum():,.2f}")
    print(f"Without duplicates: {deduplicated['Quantity'].sum():,.2f}")

    print("\n=== Gross line-value totals ===")
    print(f"With duplicates: {data['line_value'].sum():,.2f}")
    print(f"Without duplicates: {deduplicated['line_value'].sum():,.2f}")

    print("\n=== Most frequently repeated full rows ===")
    print(
        data.loc[duplicate_mask]
        .head(10)
        .drop(columns=["line_value"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
