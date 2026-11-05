"""Observational retail dataset generator for causal inference and uplift modeling."""

from typing import Optional
import numpy as np
import pandas as pd


def generate_observational_retail_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic observational retail customer data with confounding and heterogeneous treatment effects.

    Features generated:
      - customer_id: Unique string identifier
      - income: Customer annual income ($)
      - tenure_months: Customer tenure in months
      - loyalty_score: Customer loyalty / engagement metric [0, 10]
      - historical_spend: Past 12-month purchase volume ($)
      - browse_minutes: Weekly website / app engagement duration
      - treatment: Discrete promotional offer in {0, 10, 20} ($ discount)
      - conversion: Binary outcome (1 = purchased, 0 = did not purchase)
      - transaction_amount: Realized spend ($) conditional on conversion
      - true_ite: Ground truth heterogeneous treatment effect of treatment vs. control (0)
      - baseline_conversion_prob: Organic conversion probability without promotional incentive

    Parameters
    ----------
    n_samples : int, default=1000
        Number of customer records to synthesize.
    seed : int, default=42
        Random seed for reproducible dataset synthesis.

    Returns
    -------
    pd.DataFrame
        Synthesized customer dataset with observational confounding and ground truth causal metrics.
    """
    rng = np.random.default_rng(seed)

    # 1. Base Customer Features
    customer_ids = [f"cust_{i + 1:05d}" for i in range(n_samples)]

    # Income ($20k to $160k, log-normally distributed)
    income = np.clip(rng.lognormal(mean=10.8, sigma=0.45, size=n_samples), 20000.0, 180000.0)

    # Customer tenure (1 to 72 months)
    tenure_months = rng.integers(1, 73, size=n_samples)

    # Loyalty score (0.0 to 10.0, beta distributed skewed slightly right)
    loyalty_score = np.clip(rng.beta(a=3.0, b=2.5, size=n_samples) * 10.0, 0.0, 10.0)

    # Historical spend (past 12 months, correlated with income and loyalty)
    spend_noise = rng.normal(0, 50.0, size=n_samples)
    historical_spend = np.clip(
        (income / 250.0) + (loyalty_score * 35.0) + spend_noise,
        15.0,
        3500.0,
    )

    # Weekly browse minutes (5 to 120 minutes)
    browse_minutes = np.clip(
        rng.gamma(shape=3.0, scale=10.0, size=n_samples) + (loyalty_score * 2.0),
        3.0,
        150.0,
    )

    # 2. Confounder Normalization for Propensity & Outcome equations
    norm_income = (income - 55000.0) / 25000.0
    norm_loyalty = (loyalty_score - 5.0) / 2.5
    norm_spend = (historical_spend - 350.0) / 200.0
    norm_browse = (browse_minutes - 35.0) / 20.0

    # 3. Treatment Assignment Mechanism with Strong Observational Confounding
    # Marketing teams disproportionately offer discounts to higher loyalty or higher browse customers
    logit_10 = -0.3 + 0.65 * norm_loyalty + 0.45 * norm_browse - 0.20 * norm_income
    logit_20 = -0.9 + 0.85 * norm_loyalty + 0.55 * norm_spend - 0.15 * norm_income

    exp_0 = np.ones(n_samples)
    exp_10 = np.exp(logit_10)
    exp_20 = np.exp(logit_20)
    sum_exp = exp_0 + exp_10 + exp_20

    prob_0 = exp_0 / sum_exp
    prob_10 = exp_10 / sum_exp
    prob_20 = exp_20 / sum_exp

    treatment = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        treatment[i] = rng.choice([0, 10, 20], p=[prob_0[i], prob_10[i], prob_20[i]])

    # 4. Baseline Conversion Probability (Organic holdout conversion Y(0))
    # Higher baseline for loyal, high-browse customers
    base_logits = -1.2 + 0.80 * norm_loyalty + 0.50 * norm_browse + 0.40 * norm_spend
    baseline_conversion_prob = np.clip(1.0 / (1.0 + np.exp(-base_logits)), 0.05, 0.95)

    # 5. Ground Truth Heterogeneous Treatment Effect (CATE / ITE)
    # Price-sensitive customers with moderate loyalty/browse have highest uplift (Persuadables)
    # Very high loyalty already buy organically (Sure Things, near-zero uplift)
    # Low spend/disengaged are Lost Causes (near-zero uplift)
    # A small fraction with high income / low loyalty churn when spammed with coupons (Sleeping Dogs, negative uplift)
    persuadable_factor = np.exp(-0.5 * ((norm_loyalty - 0.2) ** 2 + (norm_browse - 0.3) ** 2))
    sleeping_dog_penalty = np.where((norm_income > 1.2) & (norm_loyalty < -0.8), -0.15, 0.0)

    # Uplift for Tier 10 vs Control 0
    ite_10 = 0.22 * persuadable_factor + sleeping_dog_penalty + rng.normal(0, 0.02, size=n_samples)
    ite_10 = np.clip(ite_10, -0.30, 0.50)

    # Uplift for Tier 20 vs Control 0 (monotonically higher uplift with diminishing marginal gain)
    ite_20 = 1.45 * ite_10 + 0.05 * persuadable_factor + rng.normal(0, 0.015, size=n_samples)
    ite_20 = np.clip(ite_20, -0.35, 0.70)

    # True realized ITE under the assigned treatment
    assigned_ite = np.zeros(n_samples, dtype=float)
    assigned_ite[treatment == 10] = ite_10[treatment == 10]
    assigned_ite[treatment == 20] = ite_20[treatment == 20]

    # 6. Conversion Outcome Generation Y = Y(0) + tau(T)
    p_conversion = np.clip(baseline_conversion_prob + assigned_ite, 0.0, 1.0)
    conversion = (rng.uniform(0, 1, size=n_samples) < p_conversion).astype(int)

    # 7. Realized Transaction Amount ($)
    # Conditional on converting, basket size is driven by historical spend and income
    base_basket = 25.0 + 0.12 * historical_spend + 0.0006 * income + rng.normal(0, 8.0, size=n_samples)
    base_basket = np.clip(base_basket, 15.0, 500.0)

    # If converted, record spend; if not, transaction amount is 0.0
    transaction_amount = np.where(conversion == 1, np.round(base_basket, 2), 0.0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "income": np.round(income, 2),
            "tenure_months": tenure_months,
            "loyalty_score": np.round(loyalty_score, 2),
            "historical_spend": np.round(historical_spend, 2),
            "browse_minutes": np.round(browse_minutes, 2),
            "treatment": treatment,
            "conversion": conversion,
            "transaction_amount": transaction_amount,
            "true_ite": np.round(assigned_ite, 4),
            "true_ite_10": np.round(ite_10, 4),
            "true_ite_20": np.round(ite_20, 4),
            "baseline_conversion_prob": np.round(baseline_conversion_prob, 4),
        }
    )

    return df

