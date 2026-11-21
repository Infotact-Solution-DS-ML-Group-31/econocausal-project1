
"""Build a documented positive-sales subset for descriptive analysis."""

from pathlib import Path

import pandas as pd

from econocausal.data.eda import load_retail_workbook

SPECIAL_STOCK_CODES = {"B", "M", "BANK CHARGES", "AMAZONFEE"}


def build_sales_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select ordinary positive-quantity, positive-price sales.

    This is a descriptive analysis subset, not a causal identification
    strategy. The raw input is never modified.
    """
    required = {
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()
    result["Quantity"] = pd.to_numeric(result["Quantity"], errors="coerce")
    result["Price"] = pd.to_numeric(result["Price"], errors="coerce")

    invoice = result["Invoice"].astype("string").str.strip()
    stock = (
        result["StockCode"].astype("string").str.strip().str.upper()
    )

    is_cancellation = invoice.str.startswith("C", na=False)
    positive_quantity = result["Quantity"].gt(0)
    positive_price = result["Price"].gt(0)
    ordinary_product = ~stock.isin(SPECIAL_STOCK_CODES)

    keep = (
        ~is_cancellation
        & positive_quantity
        & positive_price
        & ordinary_product
    )

    sales = result.loc[keep].copy()

    report = {
        "input_rows": int(len(result)),
        "cancellation_rows_excluded": int(is_cancellation.sum()),
        "nonpositive_quantity_rows_excluded": int(
            (~positive_quantity).sum()
        ),
        "nonpositive_price_rows_excluded": int(
            (~positive_price).sum()
        ),
        "special_stock_code_rows_excluded": int(
            (~ordinary_product).sum()
        ),
        "output_rows": int(len(sales)),
        "output_exact_duplicate_rows": int(sales.duplicated().sum()),
        "missing_customer_ids_retained": int(
            sales["Customer ID"].isna().sum()
        ),
    }

    return sales, report


def main() -> None:
    source = Path("data/raw/online_retail_II.xlsx")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_retail_workbook(source)
    sales, report = build_sales_dataset(data)

    sales_path = output_dir / "retail_sales.csv"
    report_path = output_dir / "sales_dataset_report.txt"

    sales.to_csv(sales_path, index=False)

    with report_path.open("w", encoding="utf-8") as file:
        for key, value in report.items():
            file.write(f"{key}: {value}\n")

    print("\n".join(f"{key}: {value}" for key, value in report.items()))
    print(f"Saved dataset: {sales_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
