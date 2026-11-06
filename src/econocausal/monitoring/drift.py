
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


WARNING_THRESHOLD = 0.10
DRIFT_THRESHOLD = 0.20
EPSILON = 1e-6


@dataclass
class FeatureDrift:
    feature: str
    feature_type: str
    psi: float
    status: str
    reference_missing_pct: float
    current_missing_pct: float


@dataclass
class DriftReport:
    status: str
    features_checked: int
    drifted_features: list[str]
    warning_features: list[str]
    results: list[FeatureDrift]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "features_checked": self.features_checked,
            "drifted_features": self.drifted_features,
            "warning_features": self.warning_features,
            "results": [asdict(result) for result in self.results],
        }


def _calculate_psi(
    reference_counts: np.ndarray,
    current_counts: np.ndarray,
) -> float:
    """Calculate PSI from aligned reference and current bin counts."""
    reference_pct = reference_counts / reference_counts.sum()
    current_pct = current_counts / current_counts.sum()

    reference_pct = np.clip(reference_pct, EPSILON, None)
    current_pct = np.clip(current_pct, EPSILON, None)

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def _numeric_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int,
) -> float:
    """Calculate numeric PSI using reference quantile boundaries."""
    ref = pd.to_numeric(reference, errors="coerce")
    cur = pd.to_numeric(current, errors="coerce")

    ref_values = ref.dropna().to_numpy(dtype=float)
    cur_values = cur.dropna().to_numpy(dtype=float)

    if len(ref_values) == 0 or len(cur_values) == 0:
        raise ValueError("Each dataset needs non-missing numeric values.")

    if not np.isfinite(ref_values).all() or not np.isfinite(cur_values).all():
        raise ValueError("Numeric features must not contain infinite values.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref_values, quantiles))

    if len(edges) == 1:
        # Handle a constant reference feature.
        value = edges[0]
        edges = np.array([-np.inf, value - 0.5, value + 0.5, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_bins = pd.cut(ref_values, bins=edges, labels=False)
    cur_bins = pd.cut(cur_values, bins=edges, labels=False)

    number_of_bins = len(edges) - 1

    ref_counts = np.bincount(
        ref_bins.astype(int),
        minlength=number_of_bins,
    ).astype(float)

    cur_counts = np.bincount(
        cur_bins.astype(int),
        minlength=number_of_bins,
    ).astype(float)

    # Treat missing values as an additional bin.
    ref_counts = np.append(ref_counts, ref.isna().sum())
    cur_counts = np.append(cur_counts, cur.isna().sum())

    return _calculate_psi(ref_counts, cur_counts)


def _categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Calculate categorical PSI, including missing and unseen values."""
    ref = reference.astype("string").fillna("__MISSING__")
    cur = current.astype("string").fillna("__MISSING__")

    categories = sorted(set(ref.unique()) | set(cur.unique()))

    ref_counts = (
        ref.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    cur_counts = (
        cur.value_counts()
        .reindex(categories, fill_value=0)
        .to_numpy(dtype=float)
    )

    return _calculate_psi(ref_counts, cur_counts)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    bins: int = 10,
) -> DriftReport:
    """
    Compare feature distributions between reference and current datasets.

    Numeric features use reference quantile bins.
    Other features use categorical frequency distributions.

    PSI thresholds:
      - below 0.10: no_drift
      - 0.10 to below 0.20: warning
      - 0.20 or higher: drift_detected
    """
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("Reference data must be a pandas DataFrame.")

    if not isinstance(current, pd.DataFrame):
        raise TypeError("Current data must be a pandas DataFrame.")

    if reference.empty or current.empty:
        raise ValueError("Reference and current datasets must not be empty.")

    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer greater than or equal to 2.")

    if features is None:
        features = list(reference.columns)

    if not features:
        raise ValueError("At least one feature must be selected.")

    if len(features) != len(set(features)):
        raise ValueError("Feature names must not contain duplicates.")

    missing_reference = sorted(set(features) - set(reference.columns))
    missing_current = sorted(set(features) - set(current.columns))

    if missing_reference or missing_current:
        raise ValueError(
            f"Missing features. Reference: {missing_reference}; "
            f"Current: {missing_current}"
        )

    results: list[FeatureDrift] = []

    for feature in features:
        ref = reference[feature]
        cur = current[feature]

        if pd.api.types.is_numeric_dtype(ref):
            if not pd.api.types.is_numeric_dtype(cur):
                raise ValueError(
                    f"Feature '{feature}' has inconsistent data types."
                )

            feature_type = "numeric"
            psi = _numeric_psi(ref, cur, bins)
        else:
            feature_type = "categorical"
            psi = _categorical_psi(ref, cur)

        if not np.isfinite(psi):
            raise ValueError(f"PSI for feature '{feature}' is not finite.")

        if psi >= DRIFT_THRESHOLD:
            status = "drift_detected"
        elif psi >= WARNING_THRESHOLD:
            status = "warning"
        else:
            status = "no_drift"

        results.append(
            FeatureDrift(
                feature=feature,
                feature_type=feature_type,
                psi=round(psi, 6),
                status=status,
                reference_missing_pct=round(
                    float(ref.isna().mean() * 100), 2
                ),
                current_missing_pct=round(
                    float(cur.isna().mean() * 100), 2
                ),
            )
        )

    drifted = [
        result.feature
        for result in results
        if result.status == "drift_detected"
    ]

    warnings = [
        result.feature
        for result in results
        if result.status == "warning"
    ]

    if drifted:
        overall_status = "drift_detected"
    elif warnings:
        overall_status = "warning"
    else:
        overall_status = "no_drift"

    return DriftReport(
        status=overall_status,
        features_checked=len(results),
        drifted_features=drifted,
        warning_features=warnings,
        results=results,
    )
