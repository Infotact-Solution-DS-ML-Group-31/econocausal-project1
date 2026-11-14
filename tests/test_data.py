
"""Tests for dataset loading."""

import pandas as pd
import pytest

from econocausal.data.load import load_csv
from econocausal.data.preprocess import (
    preprocess_dataframe,
    validate_dataframe,
)


def test_load_csv_reads_data(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"price": [10, 20], "sales": [3, 5]}).to_csv(
        path, index=False
    )

    result = load_csv(path)

    assert list(result.columns) == ["price", "sales"]
    assert len(result) == 2


def test_load_csv_validates_required_columns(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"price": [10]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_csv(path, required_columns=["price", "sales"])


def test_load_csv_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_csv(path)


def test_load_csv_rejects_header_only_file(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text("price,sales\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        load_csv(path)





def test_validate_dataframe_returns_copy():
    original = pd.DataFrame({"price": [10], "sales": [3]})

    result = validate_dataframe(
        original,
        required_columns=["price", "sales"],
    )

    assert result is not original
    pd.testing.assert_frame_equal(result, original)


def test_validate_dataframe_rejects_missing_columns():
    dataframe = pd.DataFrame({"price": [10]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(dataframe, required_columns=["price", "sales"])


def test_preprocess_dataframe_can_remove_duplicates():
    dataframe = pd.DataFrame(
        {"price": [10, 10, 20], "sales": [3, 3, 5]}
    )

    result = preprocess_dataframe(
        dataframe,
        required_columns=["price", "sales"],
        drop_duplicates=True,
    )

    assert len(result) == 2


def test_preprocess_dataframe_preserves_missing_values():
    dataframe = pd.DataFrame(
        {"price": [10, None], "sales": [3, 5]}
    )

    result = preprocess_dataframe(
        dataframe,
        required_columns=["price", "sales"],
    )

    assert result["price"].isna().sum() == 1
