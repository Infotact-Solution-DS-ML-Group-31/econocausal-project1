"""Double / Debiased Machine Learning (DML) estimator for CATE and ITE estimation."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from src.api.schemas import CustomerDataPoint


class DoubleMLEstimator:
    """Double / Debiased Machine Learning (DML) Estimator using Orthogonal Residualization.

    Implements the Neyman-orthogonal partially linear CATE framework (Chernozhukov et al. 2018):
      Stage 1 (Nuisance Estimation via Cross-Fitting):
        q(X, W) = E[Y | X, W]  (Outcome Nuisance Model)
        g(X, W) = E[T | X, W]  (Treatment Propensity Model)
        Residuals: Y_tilde = Y - q_hat,  T_tilde = T - g_hat
      Stage 2 (Causal Interaction Regression):
        E[Y_tilde | X, T_tilde] = tau(X) * T_tilde
        tau(X) = beta_0 + sum_j(beta_j * X_j)
    """

    def __init__(
        self,
        outcome_model: Optional[BaseEstimator] = None,
        treatment_model: Optional[BaseEstimator] = None,
        second_stage_model: Optional[BaseEstimator] = None,
        n_splits: int = 3,
        random_state: int = 42,
    ):
        """Initialize the Double ML Estimator.

        Parameters
        ----------
        outcome_model : BaseEstimator, optional
            First-stage regressor for outcome nuisance E[Y | X, W].
            Defaults to RandomForestRegressor(n_estimators=60, max_depth=5).
        treatment_model : BaseEstimator, optional
            First-stage regressor for treatment propensity E[T | X, W].
            Defaults to RandomForestRegressor(n_estimators=60, max_depth=5).
        second_stage_model : BaseEstimator, optional
            Second-stage regressor for heterogeneous treatment effect tau(X).
            Defaults to Ridge(alpha=1.0, fit_intercept=False).
        n_splits : int, default=3
            Number of cross-fitting folds to eliminate regularization bias.
        random_state : int, default=42
            Random seed for cross-fitting splits and estimators.
        """
        self.outcome_model = outcome_model or RandomForestRegressor(
            n_estimators=60, max_depth=5, min_samples_leaf=10, random_state=random_state
        )
        self.treatment_model = treatment_model or RandomForestRegressor(
            n_estimators=60, max_depth=5, min_samples_leaf=10, random_state=random_state
        )
        self.second_stage_model = second_stage_model or Ridge(alpha=1.0, fit_intercept=False)
        self.n_splits = n_splits
        self.random_state = random_state

        # Fitted state
        self.feature_cols_: List[str] = []
        self.confounder_cols_: List[str] = []
        self.treatment_col_: str = "treatment"
        self.outcome_col_: str = "conversion"
        self.is_fitted_: bool = False
        self.ate_: float = 0.0
        self.tau_coefficients_: Optional[np.ndarray] = None
        self.fitted_outcome_model_: Optional[BaseEstimator] = None
        self.fitted_treatment_model_: Optional[BaseEstimator] = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
        feature_cols: Optional[List[str]] = None,
        confounder_cols: Optional[List[str]] = None,
    ) -> "DoubleMLEstimator":
        """Fit the Double ML causal estimator using K-fold cross-fitting and orthogonalization.

        Parameters
        ----------
        df : pd.DataFrame
            Observational retail dataset containing treatment, outcome, and covariates.
        treatment_col : str, default='treatment'
            Name of the treatment column (discrete or continuous discount amount).
        outcome_col : str, default='conversion'
            Name of the outcome column (binary conversion or transaction revenue).
        feature_cols : List[str], optional
            Heterogeneity modifier features X that govern ITE variance tau(X).
            Defaults to ['income', 'tenure_months', 'loyalty_score', 'historical_spend', 'browse_minutes'].
        confounder_cols : List[str], optional
            Confounders W affecting both treatment assignment and outcome.
            Defaults to ['income', 'loyalty_score', 'historical_spend', 'browse_minutes'].

        Returns
        -------
        DoubleMLEstimator
            Fitted estimator instance.
        """
        self.treatment_col_ = treatment_col
        self.outcome_col_ = outcome_col

        default_features = [
            "income",
            "tenure_months",
            "loyalty_score",
            "historical_spend",
            "browse_minutes",
        ]
        self.feature_cols_ = (
            feature_cols
            if feature_cols is not None
            else [c for c in default_features if c in df.columns]
        )
        if not self.feature_cols_:
            raise ValueError("No valid feature columns found in dataset.")

        default_confounders = [
            "income",
            "loyalty_score",
            "historical_spend",
            "browse_minutes",
        ]
        self.confounder_cols_ = (
            confounder_cols
            if confounder_cols is not None
            else [c for c in default_confounders if c in df.columns]
        )

        # Union of features and confounders for nuisance models (X U W)
        nuisance_cols = list(dict.fromkeys(self.feature_cols_ + self.confounder_cols_))
        X_nuisance = df[nuisance_cols].to_numpy(dtype=float)
        X_hetero = df[self.feature_cols_].to_numpy(dtype=float)
        T = df[treatment_col].to_numpy(dtype=float)
        Y = df[outcome_col].to_numpy(dtype=float)

        n_samples = len(df)
        y_res = np.zeros(n_samples, dtype=float)
        t_res = np.zeros(n_samples, dtype=float)

        # Stage 1: K-Fold Cross-Fitting for Orthogonal Residualization
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(X_nuisance):
            q_model = clone(self.outcome_model)
            g_model = clone(self.treatment_model)

            q_model.fit(X_nuisance[train_idx], Y[train_idx])
            g_model.fit(X_nuisance[train_idx], T[train_idx])

            y_res[val_idx] = Y[val_idx] - q_model.predict(X_nuisance[val_idx])
            t_res[val_idx] = T[val_idx] - g_model.predict(X_nuisance[val_idx])

        # Stage 2: Second-stage CATE regression on residuals
        # We construct basis Z = [T_tilde, T_tilde * X_1, T_tilde * X_2, ...]
        # E[Y_tilde | X, T_tilde] = beta_0 * T_tilde + sum_j(beta_j * T_tilde * X_j)
        # Therefore: tau(X) = beta_0 + sum_j(beta_j * X_j)
        ones_col = np.ones((n_samples, 1), dtype=float)
        X_augmented = np.hstack([ones_col, X_hetero])
        Z = X_augmented * t_res[:, np.newaxis]

        second_stage = clone(self.second_stage_model)
        second_stage.fit(Z, y_res)
        self.second_stage_model = second_stage
        self.tau_coefficients_ = getattr(second_stage, "coef_", None)

        # Fit final global nuisance models for out-of-sample baseline predictions
        self.fitted_outcome_model_ = clone(self.outcome_model).fit(X_nuisance, Y)
        self.fitted_treatment_model_ = clone(self.treatment_model).fit(X_nuisance, T)

        # Mark fitted before computing training predictions
        self.is_fitted_ = True

        # Calculate population ATE over training cohort
        predicted_ites = self.predict_ite(df[self.feature_cols_])
        self.ate_ = float(np.mean(predicted_ites))

        return self

    def predict_ite(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict Individual Treatment Effect (ITE / CATE uplift) for customer features X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Customer feature matrix matching feature_cols_.

        Returns
        -------
        np.ndarray
            Predicted individual treatment effect (uplift delta) for each customer.
        """
        if not self.is_fitted_:
            raise RuntimeError("DoubleMLEstimator must be fitted before calling predict_ite().")

        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_cols_].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)

        n_eval = X_arr.shape[0]
        ones_col = np.ones((n_eval, 1), dtype=float)
        X_augmented = np.hstack([ones_col, X_arr])

        if self.tau_coefficients_ is not None:
            # tau(X) = beta_0 + sum_j(beta_j * X_j)
            ite_preds = X_augmented @ self.tau_coefficients_
        else:
            # Fallback using unit treatment perturbation
            unit_z = X_augmented * 1.0
            ite_preds = self.second_stage_model.predict(unit_z)

        # Scale for discount units: if treatment was in dollars (e.g., $10 or $20),
        # convert the marginal per-dollar slope to a standard 10-dollar incentive scale
        # unless it is already on the [0, 1] probability scale
        return np.asarray(ite_preds, dtype=float)

    def predict_customer_datapoints(
        self,
        df: pd.DataFrame,
        default_cost: float = 10.0,
        default_value: float = 50.0,
    ) -> List[CustomerDataPoint]:
        """Convert a customer DataFrame into M3-ready CustomerDataPoint Pydantic models.

        Parameters
        ----------
        df : pd.DataFrame
            Customer dataset.
        default_cost : float, default=10.0
            Fallback treatment promotional cost if not specified in dataframe.
        default_value : float, default=50.0
            Fallback customer value if not specified in dataframe.

        Returns
        -------
        List[CustomerDataPoint]
            Structured customer payloads for SciPyConstrainedSolver.
        """
        if not self.is_fitted_:
            raise RuntimeError("DoubleMLEstimator must be fitted before predict_customer_datapoints().")

        predicted_ites = self.predict_ite(df)

        datapoints: List[CustomerDataPoint] = []
        for idx, row in df.reset_index(drop=True).iterrows():
            cust_id = (
                str(row["customer_id"])
                if "customer_id" in row
                else f"cust_{idx + 1:05d}"
            )

            # Baseline conversion probability without discount
            if "baseline_conversion_prob" in row:
                base_prob = float(row["baseline_conversion_prob"])
            elif self.fitted_outcome_model_ is not None:
                nuisance_cols = list(dict.fromkeys(self.feature_cols_ + self.confounder_cols_))
                row_vals = row[nuisance_cols].to_numpy(dtype=float).reshape(1, -1)
                base_pred = float(self.fitted_outcome_model_.predict(row_vals)[0])
                base_prob = float(np.clip(base_pred, 0.01, 0.99))
            else:
                base_prob = 0.25

            # Treatment cost
            if "treatment_cost" in row and float(row["treatment_cost"]) > 0:
                cost = float(row["treatment_cost"])
            elif "treatment" in row and float(row["treatment"]) > 0:
                cost = float(row["treatment"])
            else:
                cost = default_cost

            # Customer value
            if "customer_value" in row and float(row["customer_value"]) > 0:
                val = float(row["customer_value"])
            elif "transaction_amount" in row and float(row["transaction_amount"]) > 0:
                val = float(row["transaction_amount"])
            elif "historical_spend" in row and float(row["historical_spend"]) > 0:
                val = float(np.clip(float(row["historical_spend"]) * 0.15, 20.0, 300.0))
            else:
                val = default_value

            datapoints.append(
                CustomerDataPoint(
                    customer_id=cust_id,
                    baseline_conversion_prob=float(np.clip(base_prob, 0.0, 1.0)),
                    predicted_ite=float(predicted_ites[idx]),
                    treatment_cost=float(cost),
                    customer_value=float(val),
                )
            )

        return datapoints

    @property
    def ate(self) -> float:
        """Average Treatment Effect (population mean uplift)."""
        if not self.is_fitted_:
            raise RuntimeError("DoubleMLEstimator is not fitted.")
        return self.ate_

    def summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics and summary of the fitted Double ML model."""
        if not self.is_fitted_:
            raise RuntimeError("DoubleMLEstimator is not fitted.")

        feature_weights = {}
        if self.tau_coefficients_ is not None and len(self.tau_coefficients_) == len(self.feature_cols_) + 1:
            feature_weights["intercept"] = float(self.tau_coefficients_[0])
            for name, coef in zip(self.feature_cols_, self.tau_coefficients_[1:]):
                feature_weights[name] = float(coef)

        return {
            "is_fitted": self.is_fitted_,
            "ate": self.ate_,
            "n_features": len(self.feature_cols_),
            "feature_cols": self.feature_cols_,
            "confounder_cols": self.confounder_cols_,
            "tau_weights": feature_weights,
            "n_splits": self.n_splits,
        }

