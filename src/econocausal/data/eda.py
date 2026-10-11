
"""Exploratory data analysis for the Online Retail II workbook."""

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def load_retail_workbook(file_path: str | Path) -> pd.DataFrame:
    """Read both sheets of the Online Retail II workbook."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")

    sheets = pd.read_excel(path, sheet_name=None)

    if not sheets:
        raise ValueError("Workbook contains no sheets")

    frames = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            continue

        missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing columns: {missing}"
            )

        frame = frame.copy()
        frame["SourceSheet"] = sheet_name
        frames.append(frame)

    if not frames:
        raise ValueError("Workbook contains no data rows")

    return pd.concat(frames, ignore_index=True)


def summarize_retail_data(dataframe: pd.DataFrame) -> dict:
    """Return core data-quality and descriptive statistics."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = dataframe.copy()

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )

    numeric_columns = ["Quantity", "Price"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invoice_text = data["Invoice"].astype("string").str.strip()

    summary = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in data.isna().sum().items()
        },
        "unique_invoices": int(data["Invoice"].nunique()),
        "unique_products": int(data["StockCode"].nunique()),
        "unique_customers": int(data["Customer ID"].nunique()),
        "date_min": (
            str(data["InvoiceDate"].min())
            if data["InvoiceDate"].notna().any()
            else None
        ),
        "date_max": (
            str(data["InvoiceDate"].max())
            if data["InvoiceDate"].notna().any()
            else None
        ),
        "negative_or_zero_quantity_rows": int(
            data["Quantity"].le(0).sum()
        ),
        "negative_or_zero_price_rows": int(data["Price"].le(0).sum()),
        "cancellation_invoice_rows": int(
            invoice_text.str.startswith("C", na=False).sum()
        ),
        "quantity_describe": data["Quantity"].describe().to_dict(),
        "price_describe": data["Price"].describe().to_dict(),
    }

    return summary


def main() -> None:
    workbook_path = Path("data/raw/online_retail_II.xlsx")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_retail_workbook(workbook_path)
    summary = summarize_retail_data(dataframe)

    summary_path = output_dir / "eda_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print(f"Rows: {summary['rows']:,}")
    print(f"Duplicate rows: {summary['duplicate_rows']:,}")
    print(f"Unique products: {summary['unique_products']:,}")
    print(f"Date range: {summary['date_min']} to {summary['date_max']}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
