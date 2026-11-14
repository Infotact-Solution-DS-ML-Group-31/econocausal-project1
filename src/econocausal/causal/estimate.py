from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression


@dataclass
class DMLEstimate:
    ate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_observations: int
    n_treated: int
    n_control: int
    propensity_min: float
    propensity_max: float
    method: str
    warnings: list[str]


def estimate_ate_dml(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    n_splits: int = 5,
    random_state: int = 42,
) -> DMLEstimate:
    """
    Estimate an ATE using cross-fitted partially linear DML.

    Treatment and outcome must be binary.
    Covariates must be numeric, complete, and measured before treatment.

    The estimator does not prove unconfoundedness, overlap,
    correct temporal ordering, or causal identification.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if not covariates:
        raise ValueError("At least one pre-treatment covariate is required.")

    required = set(covariates) | {treatment_col, outcome_col}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[list(required)].copy()

    if data.isna().any().any():
        raise ValueError("Missing values are not allowed.")

    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("All input values must be finite numeric values.")

    treatment_raw = data[treatment_col].to_numpy(dtype=float)
    outcome = data[outcome_col].to_numpy(dtype=float)
    features = data[covariates].to_numpy(dtype=float)

    if not np.isin(treatment_raw, [0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    treatment = treatment_raw.astype(int)

    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("Outcome must contain only 0 and 1.")

    n_treated = int(treatment.sum())
    n_control = len(treatment) - n_treated

    if n_treated == 0 or n_control == 0:
        raise ValueError("Both treatment groups are required.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if min(n_treated, n_control) < n_splits:
        raise ValueError(
            "Each treatment group must contain at least n_splits observations."
        )

    # Cross-fitted nuisance models.
    outcome_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )

    propensity_model = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # m(X) = E[Y | X]
    outcome_predictions = cross_val_predict(
        clone(outcome_model),
        features,
        outcome,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )

    # e(X) = P[T=1 | X]
    propensity_predictions = cross_val_predict(
        clone(propensity_model),
        features,
        treatment,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Clip only to prevent numerical instability.
    propensity_predictions = np.clip(
        propensity_predictions,
        0.01,
        0.99,
    )

    treatment_residual = treatment - propensity_predictions
    outcome_residual = outcome - outcome_predictions

    denominator = np.mean(treatment_residual**2)

    if denominator <= 1e-10:
        raise ValueError(
            "Insufficient treatment variation after conditioning on covariates."
        )

    # Partially linear DML coefficient.
    ate = float(
        np.mean(treatment_residual * outcome_residual) / denominator
    )

    # Influence-function based standard error.
    influence = (
        treatment_residual
        * (outcome_residual - ate * treatment_residual)
        / denominator
    )

    standard_error = float(
    np.std(influence, ddof=1) / np.sqrt(len(data))
)

    if not np.isfinite(standard_error):
        raise ValueError("Standard error could not be estimated.")

    critical_value = norm.ppf(0.975)

    confidence_interval = (
        ate - critical_value * standard_error,
        ate + critical_value * standard_error,
    )

    warnings = [
        "Causal interpretation requires conditional exchangeability.",
        "Positivity and overlap must be assessed separately.",
        "All covariates must be measured before treatment assignment.",
        "Cross-fitting does not eliminate unmeasured confounding.",
        "This estimator does not support the current historical retail dataset without valid treatment assignment and exposure outcomes.",
    ]

    if propensity_predictions.min() < 0.05 or propensity_predictions.max() > 0.95:
        warnings.append(
            "Potential propensity-score overlap problems detected."
        )

    return DMLEstimate(
        ate=ate,
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        n_observations=len(data),
        n_treated=n_treated,
        n_control=n_control,
        propensity_min=float(propensity_predictions.min()),
        propensity_max=float(propensity_predictions.max()),
        method="Cross-fitted partially linear Double Machine Learning",
        warnings=warnings,
    )