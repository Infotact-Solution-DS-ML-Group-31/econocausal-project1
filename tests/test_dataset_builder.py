
import pandas as pd

from econocausal.data.build_dataset import build_sales_dataset


def sample_data():
    return pd.DataFrame(
        {
            "Invoice": ["1001", "1002", "C1003", "1004", "1005", "1006"],
            "StockCode": ["A1", "A1", "A1", "A1", "B", "A2"],
            "Description": ["Item A", "Item A", "Item A", "Item A", "Manual", "Item B"],
            "Quantity": [2, -1, -2, 3, 1, 1],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-01"] * 6
            ),
            "Price": [5.0, 5.0, 5.0, 0.0, 10.0, 4.0],
            "Customer ID": [10, 10, 10, 11, 12, None],
            "Country": ["UK"] * 6,
            "SourceSheet": ["test"] * 6,
        }
    )


def test_builder_keeps_only_positive_ordinary_sales():
    result, report = build_sales_dataset(sample_data())

    assert len(result) == 2
    assert result["Invoice"].tolist() == ["1001", "1006"]
    assert (result["Quantity"] > 0).all()
    assert (result["Price"] > 0).all()
    assert report["output_rows"] == 2


def test_builder_preserves_missing_customer_ids():
    result, _ = build_sales_dataset(sample_data())

    assert result["Customer ID"].isna().sum() == 1


def test_builder_does_not_remove_duplicates():
    data = sample_data()
    duplicated_row = data.iloc[[0]]
    data = pd.concat([data, duplicated_row], ignore_index=True)

    result, report = build_sales_dataset(data)

    assert report["output_exact_duplicate_rows"] == 1
    assert len(result) == 3
