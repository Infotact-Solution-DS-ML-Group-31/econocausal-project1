"""Covariate and treatment effect drift monitoring using PSI and KS tests."""

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DriftReport:
    """Report detailing statistical distribution drift analysis."""

    drift_detected: bool
    severity: str  # "none", "moderate", or "critical"
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    recommendation: str


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate the Population Stability Index (PSI) between two 1D distributions.

    Args:
        reference: Historical or baseline 1D distribution array.
        current: Current or newly scored 1D distribution array.
        bins: Number of histogram bins to compute shared discretization.

    Returns:
        PSI metric rounded to 6 decimal places.

    Raises:
        ValueError: If inputs are empty, non-1D, or contain NaN/Inf.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)

    if ref.size == 0:
        raise ValueError("Reference distribution array cannot be empty.")
    if cur.size == 0:
        raise ValueError("Current distribution array cannot be empty.")

    if ref.ndim != 1:
        raise ValueError(f"Reference array must be 1D, got {ref.ndim}D.")
    if cur.ndim != 1:
        raise ValueError(f"Current array must be 1D, got {cur.ndim}D.")

    if np.any(np.isnan(ref)) or np.any(np.isinf(ref)):
        raise ValueError("Reference array contains NaN or Inf values.")
    if np.any(np.isnan(cur)) or np.any(np.isinf(cur)):
        raise ValueError("Current array contains NaN or Inf values.")

    if bins <= 0:
        raise ValueError(f"Number of bins must be positive, got {bins}.")

    min_val = min(float(ref.min()), float(cur.min()))
    max_val = max(float(ref.max()), float(cur.max()))

    if np.isclose(min_val, max_val):
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    # Apply epsilon clipping (1e-6) to bin proportions
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(0.0, psi_val), 6)


def evaluate_distribution_drift(
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
) -> DriftReport:
    """Evaluate statistical distribution drift using PSI and Kolmogorov-Smirnov test.

    Args:
        reference: Baseline reference distribution array.
        current: Current observation distribution array.
        psi_threshold: Threshold above which PSI triggers critical severity (default 0.25).
        ks_alpha: Significance level alpha for the 2-sample KS test (default 0.05).

    Returns:
        DriftReport with drift flag, severity classification, test statistics, and recommendation.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)

    psi_score = calculate_psi(ref, cur)
    ks_res = stats.ks_2samp(ref, cur)
    ks_statistic = float(ks_res.statistic)
    ks_pvalue = float(ks_res.pvalue)

    if psi_score >= psi_threshold or (ks_pvalue < ks_alpha and psi_score >= 0.10):
        severity = "critical"
    elif 0.10 <= psi_score < psi_threshold:
        severity = "moderate"
    else:
        severity = "none"

    drift_detected = severity in ("moderate", "critical")

    if severity == "critical":
        recommendation = (
            "Critical distribution shift detected; retrain EconML Double ML model "
            "before allocating budget."
        )
    elif severity == "moderate":
        recommendation = (
            "Moderate distribution drift detected; monitor allocation metrics "
            "and schedule model retraining."
        )
    else:
        recommendation = (
            "No significant distribution drift detected; safe to proceed with budget allocation."
        )

    return DriftReport(
        drift_detected=drift_detected,
        severity=severity,
        psi_score=psi_score,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        recommendation=recommendation,
    )
