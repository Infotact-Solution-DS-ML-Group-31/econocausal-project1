
"""Data validation and preprocessing utilities."""

import pandas as pd


def validate_dataframe(
    dataframe: pd.DataFrame,
    *,
    required_columns: list[str],
) -> pd.DataFrame:
    """Validate that a dataset contains required columns and data rows."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError("Dataset contains no data rows")

    if dataframe.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")

    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return dataframe.copy()



def preprocess_dataframe(
    dataframe: pd.DataFrame,
    *,
    required_columns: list[str],
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Validate a dataset and apply basic, configurable cleaning.

    Missing values are preserved rather than filled automatically because
    imputation decisions should depend on the meaning of each variable.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    if dataframe.empty:
        raise ValueError("Dataset contains no data rows")

    result = dataframe.copy()
    result.columns = [str(column).strip() for column in result.columns]

    if any(not column for column in result.columns):
        raise ValueError("Dataset contains an empty column name")

    if result.columns.duplicated().any():
        raise ValueError("Column names are duplicated after normalization")

    result = validate_dataframe(
        result,
        required_columns=required_columns,
    )

    if drop_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)

    return result
