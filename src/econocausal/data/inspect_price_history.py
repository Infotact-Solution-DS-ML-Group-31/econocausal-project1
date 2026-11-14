
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "data" / "processed" / "monthly_product_prices.csv"

CANDIDATE_CODES = ["85099B", "47566", "20685", "48138"]


def main():
    df = pd.read_csv(INPUT)

    for code in CANDIDATE_CODES:
        product = df[df["StockCode"].astype(str) == code].sort_values("Month")

        print(f"\n=== Product {code} ===")
        if product.empty:
            print("No records found.")
            continue

        print(
            product[
                [
                    "Month",
                    "transaction_rows",
                    "median_price",
                    "distinct_prices",
                    "total_quantity",
                ]
            ].to_string(index=False)
        )

        prices = product["median_price"]
        changes = prices.diff().dropna()
        changed_months = int((changes != 0).sum())

        print(f"Active months: {len(product)}")
        print(f"Months with a median-price change: {changed_months}")
        print(f"Minimum monthly median: {prices.min():.4f}")
        print(f"Maximum monthly median: {prices.max():.4f}")


if __name__ == "__main__":
    main()
