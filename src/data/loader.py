"""Data loader and validator for raw retail transaction and campaign datasets."""

import json
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd


class RetailDataLoader:
    """Loads, validates, coerces, and cleans retail customer campaign datasets.

    Enforces data integrity, schema consistency, missing value imputation,
    and deduplication across CSV, JSON, and DataFrame inputs.
    """

    REQUIRED_COLUMNS: List[str] = [
        "customer_id",
        "income",
        "tenure_months",
        "loyalty_score",
        "historical_spend",
        "browse_minutes",
        "treatment",
        "conversion",
        "transaction_amount",
    ]

    NUMERICAL_BOUNDS = {
        "income": (0.0, None),
        "tenure_months": (0.0, None),
        "loyalty_score": (0.0, None),
        "historical_spend": (0.0, None),
        "browse_minutes": (0.0, None),
        "treatment": (0.0, None),
        "conversion": (0.0, 1.0),
        "transaction_amount": (0.0, None),
    }

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        drop_duplicates: bool = True,
        impute_missing: bool = True,
    ):
        """Initialize RetailDataLoader.

        Parameters
        ----------
        required_columns : List[str], optional
            List of required column headers. Defaults to REQUIRED_COLUMNS.
        drop_duplicates : bool, default=True
            Whether to drop duplicate customer_id records.
        impute_missing : bool, default=True
            Whether to impute missing values (median for continuous, mode for discrete).
        """
        self.required_columns = required_columns or list(self.REQUIRED_COLUMNS)
        self.drop_duplicates = drop_duplicates
        self.impute_missing = impute_missing

    def load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load and process a dataset from a file path (CSV or JSON) or existing DataFrame.

        Parameters
        ----------
        source : str, Path, or pd.DataFrame
            Source dataset path or pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned, typed, and validated customer dataset.
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Source file not found at: {path}")

            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix == ".json":
                df = pd.read_json(path)
            else:
                # Try reading as CSV first, then JSON
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_json(path)
        else:
            raise TypeError(
                f"Unsupported source type: {type(source)}. Expected str, Path, or pd.DataFrame."
            )

        return self.process(df)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, coerce types, impute missing values, and deduplicate dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned and validated DataFrame.
        """
        if df.empty:
            raise ValueError("Provided dataset is empty.")

        # 1. Validate required columns
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Dataset missing required columns: {missing_cols}. Present: {list(df.columns)}"
            )

        cleaned_df = df.copy()

        # 2. Deduplicate customer_id
        if self.drop_duplicates and "customer_id" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=["customer_id"], keep="first")

        cleaned_df["customer_id"] = cleaned_df["customer_id"].astype(str)

        # 3. Coerce numerical types
        numerical_cols = [
            "income",
            "tenure_months",
            "loyalty_score",
            "historical_spend",
            "browse_minutes",
            "treatment",
            "conversion",
            "transaction_amount",
        ]

        for col in numerical_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

        # 4. Impute missing values
        if self.impute_missing:
            # Discrete / categorical columns -> mode
            for col in ["treatment", "conversion"]:
                if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                    mode_val = cleaned_df[col].mode(dropna=True)
                    fill_val = mode_val.iloc[0] if not mode_val.empty else 0
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)

            # Continuous columns -> median
            continuous_cols = [
                "income",
                "tenure_months",
                "loyalty_score",
                "historical_spend",
                "browse_minutes",
                "transaction_amount",
            ]
            for col in continuous_cols:
                if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                    median_val = cleaned_df[col].median(skipna=True)
                    if pd.isna(median_val):
                        median_val = 0.0
                    cleaned_df[col] = cleaned_df[col].fillna(median_val)

        # 5. Enforce boundaries
        for col, (lower, upper) in self.NUMERICAL_BOUNDS.items():
            if col in cleaned_df.columns:
                if lower is not None:
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower)
                if upper is not None:
                    cleaned_df[col] = cleaned_df[col].clip(upper=upper)

        # Discrete integrity
        if "treatment" in cleaned_df.columns:
            cleaned_df["treatment"] = cleaned_df["treatment"].round().astype(int)
        if "conversion" in cleaned_df.columns:
            cleaned_df["conversion"] = cleaned_df["conversion"].round().astype(int)
        if "tenure_months" in cleaned_df.columns:
            cleaned_df["tenure_months"] = cleaned_df["tenure_months"].round().astype(int)

        return cleaned_df.reset_index(drop=True)

