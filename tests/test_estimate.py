import numpy as np
import pandas as pd
import pytest

from econocausal.causal.estimate import estimate_ate_dml


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 3000

    age = rng.normal(40, 10, n)
    prior_purchases = rng.poisson(3, n)

    propensity = 1 / (
        1 + np.exp(-(-0.3 + 0.015 * (age - 40) + 0.1 * prior_purchases))
    )

    treatment = rng.binomial(1, propensity)

    probability = 1 / (
        1 + np.exp(
            -(-1.0 + 0.025 * (age - 40)
              + 0.15 * prior_purchases + 0.6 * treatment)
        )
    )

    outcome = rng.binomial(1, probability)

    return pd.DataFrame({
        "age": age,
        "prior_purchases": prior_purchases,
        "treatment": treatment,
        "outcome": outcome,
    })


def test_dml_returns_valid_estimate(synthetic_data):
    result = estimate_ate_dml(
        synthetic_data,
        covariates=["age", "prior_purchases"],
    )

    assert np.isfinite(result.ate)
    assert result.standard_error > 0
    assert result.n_observations == len(synthetic_data)
    assert result.n_treated > 0
    assert result.n_control > 0


def test_dml_estimate_is_reasonably_close_to_simulated_effect(synthetic_data):
    result = estimate_ate_dml(
        synthetic_data,
        covariates=["age", "prior_purchases"],
    )

    # The simulated treatment effect is positive.
    assert result.ate > 0
    assert result.ate < 0.5


def test_dml_rejects_missing_covariate(synthetic_data):
    with pytest.raises(ValueError, match="Missing required columns"):
        estimate_ate_dml(
            synthetic_data,
            covariates=["missing_feature"],
        )


def test_dml_rejects_missing_values(synthetic_data):
    data = synthetic_data.copy()
    data.loc[0, "age"] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        estimate_ate_dml(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_dml_rejects_non_binary_treatment(synthetic_data):
    data = synthetic_data.copy()
    data["treatment"] = data["treatment"].astype(int)
    data.loc[0, "treatment"] = 2

    with pytest.raises(ValueError, match="Treatment must contain"):
        estimate_ate_dml(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_dml_requires_both_treatment_groups(synthetic_data):
    data = synthetic_data.copy()
    data["treatment"] = 1

    with pytest.raises(ValueError, match="Both treatment groups"):
        estimate_ate_dml(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_dml_rejects_too_many_folds(synthetic_data):
    data = synthetic_data.iloc[:20].copy()
    data["treatment"] = 0
    data.loc[:1, "treatment"] = 1

    with pytest.raises(ValueError, match="at least n_splits"):
        estimate_ate_dml(
            data,
            covariates=["age", "prior_purchases"],
            n_splits=5,
        )