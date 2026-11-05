"""Exploratory Data Analysis (EDA) and causal profiling for retail promotional datasets."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive exploratory data analysis, stratification, and distributional diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset containing covariates, treatments, and conversion outcomes.

    Returns
    -------
    Dict[str, Any]
        Dictionary detailing stratified metrics, treatment-control contrasts,
        percentiles, and skewness/kurtosis diagnostics.
    """
    if df.empty:
        raise ValueError("Input DataFrame for EDA is empty.")

    total_customers = len(df)
    treatment_col = "treatment" if "treatment" in df.columns else None
    conversion_col = "conversion" if "conversion" in df.columns else None
    spend_col = "transaction_amount" if "transaction_amount" in df.columns else None

    # 1. Overall Population Summary
    overall_cr = float(df[conversion_col].mean()) if conversion_col else 0.0
    total_conversions = int(df[conversion_col].sum()) if conversion_col else 0
    total_revenue = float(df[spend_col].sum()) if spend_col else 0.0

    report: Dict[str, Any] = {
        "overview": {
            "total_customers": total_customers,
            "total_conversions": total_conversions,
            "overall_conversion_rate": overall_cr,
            "total_revenue": total_revenue,
        },
        "stratified_by_treatment": {},
        "treatment_contrasts": {},
        "feature_distributions": {},
        "distribution_diagnostics": {},
    }

    # 2. Stratification by Treatment Tier ($0, $10, $20)
    if treatment_col and conversion_col:
        unique_tiers = sorted(df[treatment_col].unique())
        control_cr = 0.0
        control_spend = 0.0

        for tier in unique_tiers:
            sub = df[df[treatment_col] == tier]
            count = len(sub)
            cr = float(sub[conversion_col].mean()) if count > 0 else 0.0
            avg_spend = float(sub[spend_col].mean()) if spend_col and count > 0 else 0.0
            avg_income = float(sub["income"].mean()) if "income" in sub.columns else 0.0
            avg_loyalty = float(sub["loyalty_score"].mean()) if "loyalty_score" in sub.columns else 0.0
            avg_hist_spend = (
                float(sub["historical_spend"].mean()) if "historical_spend" in sub.columns else 0.0
            )

            report["stratified_by_treatment"][f"tier_{tier}"] = {
                "treatment_value": int(tier),
                "customer_count": count,
                "percentage_of_cohort": float(count / total_customers),
                "conversion_rate": cr,
                "average_transaction_amount": avg_spend,
                "average_income": avg_income,
                "average_loyalty_score": avg_loyalty,
                "average_historical_spend": avg_hist_spend,
            }

            if tier == 0:
                control_cr = cr
                control_spend = avg_spend

        # 3. Treatment vs Control Observational Contrasts
        for tier in unique_tiers:
            if tier != 0:
                tier_info = report["stratified_by_treatment"][f"tier_{tier}"]
                cr_delta = tier_info["conversion_rate"] - control_cr
                spend_delta = tier_info["average_transaction_amount"] - control_spend
                report["treatment_contrasts"][f"tier_{tier}_vs_control"] = {
                    "treatment_tier": int(tier),
                    "raw_conversion_rate_lift": float(cr_delta),
                    "relative_conversion_lift": float(cr_delta / control_cr) if control_cr > 0 else 0.0,
                    "raw_spend_delta": float(spend_delta),
                }

    # 4. Feature Summary & Distributional Diagnostics (Skewness & Kurtosis)
    numerical_features = [
        "income",
        "tenure_months",
        "loyalty_score",
        "historical_spend",
        "browse_minutes",
    ]

    for col in numerical_features:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                col_skew = float(stats.skew(series))
                col_kurt = float(stats.kurtosis(series))

                report["feature_distributions"][col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "p25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "p75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                }

                report["distribution_diagnostics"][col] = {
                    "skewness": col_skew,
                    "kurtosis": col_kurt,
                    "is_highly_skewed": bool(abs(col_skew) > 1.0),
                    "heavy_tailed": bool(col_kurt > 3.0),
                }

    return report

