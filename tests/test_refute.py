import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

import econocausal.causal.refute as refute


def sample_data():
    return pd.DataFrame({
        "customer_id": range(10),
        "treatment": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "outcome": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        "age": [22, 25, 31, 28, 35, 24, 29, 33, 26, 30],
    })


def test_placebo_test_returns_expected_result(monkeypatch):
    calls = []

    def fake_estimator(**kwargs):
        calls.append(kwargs["df"])
        return SimpleNamespace(ate=0.2 if len(calls) == 1 else 0.01)

    monkeypatch.setattr(refute, "estimate_ate_dml", fake_estimator)

    df = sample_data()

    result = refute.placebo_treatment_test(
        df,
        covariates=["age"],
        n_repeats=3,
    )

    assert result.observed_effect == 0.2
    assert len(result.placebo_effects) == 3
    assert result.placebo_effects == (0.01, 0.01, 0.01)
    assert result.fraction_at_least_as_extreme == 0.0
    assert result.n_repeats == 3


def test_placebo_test_preserves_treated_count_and_original_data(monkeypatch):
    treatment_counts = []

    def fake_estimator(**kwargs):
        treatment_counts.append(
            int(kwargs["df"]["treatment"].sum())
        )
        return SimpleNamespace(ate=0.1)

    monkeypatch.setattr(refute, "estimate_ate_dml", fake_estimator)

    df = sample_data()
    original = df.copy(deep=True)

    refute.placebo_treatment_test(
        df,
        covariates=["age"],
        n_repeats=4,
    )

    assert treatment_counts == [5, 5, 5, 5, 5]
    pd.testing.assert_frame_equal(df, original)


def test_placebo_test_rejects_zero_repeats():
    with pytest.raises(ValueError, match="n_repeats"):
        refute.placebo_treatment_test(
            sample_data(),
            covariates=["age"],
            n_repeats=0,
        )


def test_placebo_test_rejects_missing_treatment_column():
    df = sample_data().drop(columns=["treatment"])

    with pytest.raises(ValueError, match="Missing treatment column"):
        refute.placebo_treatment_test(
            df,
            covariates=["age"],
        )