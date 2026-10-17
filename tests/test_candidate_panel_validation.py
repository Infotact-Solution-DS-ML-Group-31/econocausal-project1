import pandas as pd
import pytest

from econocausal.data.validate_candidate_panel import validate_panel


def sample_panel():
    return pd.DataFrame(
        {
            "StockCode": ["A1", "A1", "A1"],
            "Month": ["2011-01", "2011-02", "2011-03"],
            "transaction_rows": [10, 12, 8],
            "median_unit_price": [5.0, 5.5, 5.5],
            "total_quantity_sold": [20, 25, 18],
            "gross_line_value": [100.0, 137.5, 99.0],
        }
    )


def test_valid_panel_passes():
    validate_panel(sample_panel())


def test_duplicate_product_month_raises_error():
    panel = sample_panel()
    panel.loc[1, "Month"] = "2011-01"

    with pytest.raises(ValueError, match="duplicate product-month keys"):
        validate_panel(panel)


def test_missing_calendar_month_raises_error():
    panel = sample_panel().drop(index=1)

    with pytest.raises(ValueError, match="missing calendar months"):
        validate_panel(panel)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("transaction_rows", 0),
        ("median_unit_price", -1.0),
        ("total_quantity_sold", None),
        ("gross_line_value", 0.0),
    ],
)
def test_invalid_numeric_values_raise_error(column, value):
    panel = sample_panel()
    panel.loc[0, column] = value

    with pytest.raises(ValueError, match="Invalid values"):
        validate_panel(panel)