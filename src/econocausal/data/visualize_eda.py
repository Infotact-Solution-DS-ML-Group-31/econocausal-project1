
"""Generate descriptive EDA charts for the retail sales subset."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("data/processed/retail_sales.csv")
OUTPUT_DIR = Path("docs/figures")


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Analysis dataset not found: {INPUT_PATH}. "
            "Run python -m econocausal.data.build_dataset first."
        )

    data = pd.read_csv(INPUT_PATH, parse_dates=["InvoiceDate"])
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

    data = data.dropna(subset=["InvoiceDate", "Quantity", "Price"]).copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Monthly gross line value.
    data["LineValue"] = data["Quantity"] * data["Price"]
    monthly = (
        data.set_index("InvoiceDate")["LineValue"]
        .resample("MS")
        .sum()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.5)
    ax.set_title("Monthly Gross Line Value")
    ax.set_xlabel("Month")
    ax.set_ylabel("Gross line value (price units)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_gross_line_value.png", dpi=150)
    plt.close(fig)

    # 2. Unit-price distribution: zoom into the 99th percentile.
    # This improves readability; the full data is retained.
    positive_prices = data.loc[data["Price"] > 0, "Price"]
    price_cap = positive_prices.quantile(0.99)
    displayed_prices = positive_prices[positive_prices <= price_cap]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(displayed_prices, bins=60)
    ax.set_title("Unit-Price Distribution (up to the 99th percentile)")
    ax.set_xlabel("Unit price")
    ax.set_ylabel("Transaction rows")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "unit_price_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Quantity distribution: focus on the central 99% of observations.
    positive_quantities = data.loc[data["Quantity"] > 0, "Quantity"]
    quantity_cap = positive_quantities.quantile(0.99)
    displayed_quantities = positive_quantities[
        positive_quantities <= quantity_cap
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(displayed_quantities, bins=60)
    ax.set_title("Quantity Distribution (up to the 99th percentile)")
    ax.set_xlabel("Quantity per transaction line")
    ax.set_ylabel("Transaction rows")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "quantity_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Rows analyzed: {len(data):,}")
    print(f"Monthly periods: {len(monthly)}")
    print(f"Price 99th percentile: {price_cap:.2f}")
    print(f"Quantity 99th percentile: {quantity_cap:.2f}")
    print(f"Charts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
