"""Preprocessor for causal modeling, feature scaling, and feature engineering."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class CausalDataPreprocessor:
    """Preprocesses customer transaction data for causal inference pipelines.

    Performs:
      1. Feature Engineering:
         - spend_per_month = historical_spend / (tenure_months + 1)
         - engagement_index = browse_minutes * loyalty_score
      2. Continuous Feature Standardization:
         - z-score scaling: (X - mean) / std for income, historical_spend, browse_minutes.
      3. Matrix extraction:
         - Returns covariate matrix X, treatment vector T, and outcome vector Y.
    """

    DEFAULT_NORM_COLS: List[str] = [
        "income",
        "historical_spend",
        "browse_minutes",
    ]

    COVARIATE_COLUMNS: List[str] = [
        "income",
        "tenure_months",
        "loyalty_score",
        "historical_spend",
        "browse_minutes",
        "spend_per_month",
        "engagement_index",
    ]

    def __init__(
        self,
        normalize_cols: Optional[List[str]] = None,
        covariate_cols: Optional[List[str]] = None,
    ):
        """Initialize CausalDataPreprocessor.

        Parameters
        ----------
        normalize_cols : List[str], optional
            Columns to standardize via z-score. Defaults to ['income', 'historical_spend', 'browse_minutes'].
        covariate_cols : List[str], optional
            Columns to include in covariate matrix X. Defaults to COVARIATE_COLUMNS.
        """
        self.normalize_cols = normalize_cols or list(self.DEFAULT_NORM_COLS)
        self.covariate_cols = covariate_cols or list(self.COVARIATE_COLUMNS)

        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific causal interaction ratios and indices.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned customer dataset.

        Returns
        -------
        pd.DataFrame
            Dataset augmented with spend_per_month and engagement_index.
        """
        df_out = df.copy()

        # Spend velocity: spend per month of tenure
        if "historical_spend" in df_out.columns and "tenure_months" in df_out.columns:
            df_out["spend_per_month"] = df_out["historical_spend"] / (
                df_out["tenure_months"] + 1.0
            )

        # Engagement index: interaction between browsing activity and loyalty
        if "browse_minutes" in df_out.columns and "loyalty_score" in df_out.columns:
            df_out["engagement_index"] = (
                df_out["browse_minutes"] * df_out["loyalty_score"]
            )

        return df_out

    def fit(self, df: pd.DataFrame) -> "CausalDataPreprocessor":
        """Compute standardization statistics (mean, std) on training dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Training customer dataset.

        Returns
        -------
        CausalDataPreprocessor
            Fitted preprocessor.
        """
        engineered = self.engineer_features(df)

        for col in self.normalize_cols:
            if col in engineered.columns:
                mean_val = float(engineered[col].mean())
                std_val = float(engineered[col].std())
                if std_val < 1e-8 or np.isnan(std_val):
                    std_val = 1.0
                self.means_[col] = mean_val
                self.stds_[col] = std_val

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and continuous standardization.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to transform.

        Returns
        -------
        pd.DataFrame
            Transformed dataset with normalized continuous features and engineered indices.
        """
        if not self.is_fitted_:
            raise RuntimeError("CausalDataPreprocessor must be fitted before calling transform().")

        transformed = self.engineer_features(df)

        for col in self.normalize_cols:
            if col in transformed.columns and col in self.means_:
                transformed[col] = (
                    transformed[col] - self.means_[col]
                ) / self.stds_[col]

        return transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform dataset in a single call.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Fully preprocessed dataset.
        """
        return self.fit(df).transform(df)

    def prepare_matrices(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract clean covariate matrix X, treatment vector T, and outcome vector Y.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or partially cleaned customer dataset.
        treatment_col : str, default='treatment'
            Treatment variable column.
        outcome_col : str, default='conversion'
            Outcome variable column.
        fit : bool, default=True
            Whether to fit the preprocessor on df before transforming.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, pd.Series]
            (X, T, Y) where X is covariates DataFrame, T is treatment Series, and Y is outcome Series.
        """
        transformed = self.fit_transform(df) if fit else self.transform(df)

        available_covariates = [c for c in self.covariate_cols if c in transformed.columns]
        X = transformed[available_covariates].copy()

        if treatment_col not in transformed.columns:
            raise KeyError(f"Treatment column '{treatment_col}' not found in dataset.")
        T = transformed[treatment_col].copy()

        if outcome_col not in transformed.columns:
            raise KeyError(f"Outcome column '{outcome_col}' not found in dataset.")
        Y = transformed[outcome_col].copy()

        return X, T, Y

