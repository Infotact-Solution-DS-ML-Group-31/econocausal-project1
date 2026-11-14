import numpy as np
import pytest

from econocausal.causal.uplift import evaluate_uplift


def sample_data():
    treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0],
    dtype=float,)
    outcome = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    predicted_uplift = np.array([
        0.9, 0.8, 0.7, 0.6,
        0.4, 0.3, 0.2, 0.1
    ])

    return treatment, outcome, predicted_uplift


def test_evaluate_uplift_returns_metrics():
    result = evaluate_uplift(*sample_data())

    assert np.isfinite(result.auuc)
    assert np.isfinite(result.qini_coefficient)
    assert result.n_observations == 8
    assert result.n_treated == 4
    assert result.n_control == 4


def test_uplift_curve_includes_origin():
    result = evaluate_uplift(*sample_data())

    assert len(result.curve) == 9
    assert result.curve[0] == 0.0
    assert result.random_curve[0] == 0.0


def test_uplift_curve_ranks_highest_scores_first():
    result = evaluate_uplift(*sample_data())

    # All first four ranked observations are treated.
    assert result.curve[1] == 1.0
    assert result.curve[2] == 2.0


def test_evaluate_uplift_rejects_unequal_lengths():
    treatment, outcome, predicted = sample_data()

    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_uplift(
            treatment,
            outcome[:-1],
            predicted,
        )


def test_evaluate_uplift_rejects_non_binary_treatment():
    treatment, outcome, predicted = sample_data()
    treatment[0] = 0.5

    with pytest.raises(ValueError, match="Treatment"):
        evaluate_uplift(treatment, outcome, predicted)


def test_evaluate_uplift_rejects_missing_treatment_group():
    treatment, outcome, predicted = sample_data()
    treatment[:] = 1

    with pytest.raises(ValueError, match="Both treatment groups"):
        evaluate_uplift(treatment, outcome, predicted)


def test_evaluate_uplift_rejects_non_finite_predictions():
    treatment, outcome, predicted = sample_data()
    predicted[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        evaluate_uplift(treatment, outcome, predicted)


def test_evaluate_uplift_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        evaluate_uplift(
            np.array([]),
            np.array([]),
            np.array([]),
        )