
"""Utilities for loading tabular datasets."""

from pathlib import Path

import pandas as pd


def load_csv(
    file_path: str | Path,
    *,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file and optionally validate required columns.

    Args:
        file_path: Path to the CSV file.
        required_columns: Column names that must exist in the dataset.

    Returns:
        The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or required columns are missing.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {path}") from exc

    if dataframe.empty:
        raise ValueError(f"Dataset contains no data rows: {path}")

    if required_columns:
        missing_columns = sorted(set(required_columns) - set(dataframe.columns))
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {missing_columns}"
            )

    return dataframe
