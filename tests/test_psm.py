
import numpy as np
import pandas as pd
import pytest

from econocausal.causal.psm import estimate_propensity_scores
from econocausal.causal.psm import check_common_support
from econocausal.causal.psm import nearest_neighbour_match
from econocausal.causal.psm import standardized_mean_differences
from econocausal.causal.psm import evaluate_matched_sample
from econocausal.causal.psm import MatchResult


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 1000

    age = rng.normal(40, 10, n)
    prior_purchases = rng.poisson(3, n)

    propensity = 1 / (
        1 + np.exp(
            -(-0.3 + 0.015 * (age - 40) + 0.1 * prior_purchases)
        )
    )

    treatment = rng.binomial(1, propensity)

    return pd.DataFrame({
        "age": age,
        "prior_purchases": prior_purchases,
        "treatment": treatment,
    })


def test_propensity_scores_are_valid(synthetic_data):
    result = estimate_propensity_scores(
        synthetic_data,
        covariates=["age", "prior_purchases"],
    )

    assert len(result.propensity_scores) == len(synthetic_data)
    assert np.isfinite(result.propensity_scores).all()
    assert np.all(result.propensity_scores > 0)
    assert np.all(result.propensity_scores < 1)


def test_propensity_result_contains_group_counts(synthetic_data):
    result = estimate_propensity_scores(
        synthetic_data,
        covariates=["age", "prior_purchases"],
    )

    assert result.n_observations == len(synthetic_data)
    assert result.n_treated == synthetic_data["treatment"].sum()
    assert result.n_control == (
        len(synthetic_data) - synthetic_data["treatment"].sum()
    )


def test_propensity_model_is_fitted(synthetic_data):
    result = estimate_propensity_scores(
        synthetic_data,
        covariates=["age", "prior_purchases"],
    )

    assert hasattr(result.model, "coef_")
    assert result.model.n_features_in_ == 2


def test_rejects_missing_covariate(synthetic_data):
    with pytest.raises(ValueError, match="Missing required columns"):
        estimate_propensity_scores(
            synthetic_data,
            covariates=["missing_feature"],
        )


def test_rejects_missing_values(synthetic_data):
    data = synthetic_data.copy()
    data.loc[0, "age"] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        estimate_propensity_scores(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_rejects_non_binary_treatment(synthetic_data):
    data = synthetic_data.copy()
    data.loc[0, "treatment"] = 2

    with pytest.raises(ValueError, match="Treatment must contain"):
        estimate_propensity_scores(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_rejects_missing_treatment_group(synthetic_data):
    data = synthetic_data.copy()
    data["treatment"] = 1

    with pytest.raises(ValueError, match="Both treatment groups"):
        estimate_propensity_scores(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_rejects_duplicate_covariates(synthetic_data):
    with pytest.raises(ValueError, match="duplicates"):
        estimate_propensity_scores(
            synthetic_data,
            covariates=["age", "age"],
        )


def test_rejects_treatment_as_covariate(synthetic_data):
    with pytest.raises(ValueError, match="Treatment cannot"):
        estimate_propensity_scores(
            synthetic_data,
            covariates=["age", "treatment"],
        )


def test_rejects_non_finite_covariates(synthetic_data):
    data = synthetic_data.copy()
    data.loc[0, "age"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        estimate_propensity_scores(
            data,
            covariates=["age", "prior_purchases"],
        )


def test_rejects_invalid_max_iter(synthetic_data):
    with pytest.raises(ValueError, match="max_iter"):
        estimate_propensity_scores(
            synthetic_data,
            covariates=["age", "prior_purchases"],
            max_iter=0,
        )

def test_common_support_identifies_overlapping_range():
    propensity = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    treatment = np.array([0, 0, 1, 1, 0, 1])

    result = check_common_support(propensity, treatment)

    assert result.lower_bound == pytest.approx(0.3)
    assert result.upper_bound == pytest.approx(0.5)
    assert result.n_in_support == 3
    assert result.n_outside_support == 3
    assert result.n_treated_in_support == 2
    assert result.n_control_in_support == 1


def test_common_support_returns_correct_mask():
    propensity = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    treatment = np.array([0, 0, 1, 1, 0, 1])

    result = check_common_support(propensity, treatment)

    expected = np.array([False, False, True, True, True, False])

    np.testing.assert_array_equal(result.in_support, expected)


def test_common_support_rejects_non_overlapping_groups():
    propensity = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    treatment = np.array([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="No common support"):
        check_common_support(propensity, treatment)


def test_common_support_rejects_invalid_propensity_scores():
    propensity = np.array([0.0, 0.2, 0.5, 0.8])
    treatment = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        check_common_support(propensity, treatment)


def test_common_support_rejects_mismatched_lengths():
    propensity = np.array([0.2, 0.4, 0.6])
    treatment = np.array([0, 1])

    with pytest.raises(ValueError, match="equal lengths"):
        check_common_support(propensity, treatment)


def test_common_support_requires_both_groups():
    propensity = np.array([0.2, 0.4, 0.6])
    treatment = np.array([1, 1, 1])

    with pytest.raises(ValueError, match="Both treatment groups"):
        check_common_support(propensity, treatment)


def test_nearest_neighbour_matches_closest_controls():
    propensity = np.array([0.20, 0.30, 0.25, 0.35])
    treatment = np.array([1, 1, 0, 0])

    result = nearest_neighbour_match(
        propensity,
        treatment,
        replacement=False,
    )

    assert result.n_matched == 2
    np.testing.assert_array_equal(result.treated_indices, [0, 1])
    np.testing.assert_array_equal(result.control_indices, [2, 3])
    np.testing.assert_allclose(result.distances, [0.05, 0.05])


def test_matching_without_replacement_uses_unique_controls():
    propensity = np.array([0.20, 0.21, 0.19, 0.22])
    treatment = np.array([1, 1, 0, 0])

    result = nearest_neighbour_match(
        propensity,
        treatment,
        replacement=False,
    )

    assert result.n_matched == 2
    assert result.n_unique_controls_used == 2
    assert len(np.unique(result.control_indices)) == 2


def test_matching_with_replacement_can_reuse_controls():
    propensity = np.array([0.20, 0.21, 0.19])
    treatment = np.array([1, 1, 0])

    result = nearest_neighbour_match(
        propensity,
        treatment,
        replacement=True,
    )

    assert result.n_matched == 2
    assert result.n_unique_controls_used == 1
    np.testing.assert_array_equal(result.control_indices, [2, 2])


def test_matching_rejects_pairs_outside_caliper():
    propensity = np.array([0.20, 0.80, 0.21, 0.79])
    treatment = np.array([1, 1, 0, 0])

    result = nearest_neighbour_match(
        propensity,
        treatment,
        caliper=0.005,
    )

    assert result.n_matched == 0
    assert result.n_unmatched_treated == 2


def test_matching_respects_common_support_mask():
    propensity = np.array([0.10, 0.20, 0.30, 0.40])
    treatment = np.array([1, 1, 0, 0])
    support = np.array([False, True, True, False])

    result = nearest_neighbour_match(
        propensity,
        treatment,
        in_support=support,
    )

    assert result.n_matched == 1
    assert result.treated_indices[0] == 1
    assert result.control_indices[0] == 2


def test_matching_reports_unmatched_treated_observations():
    propensity = np.array([0.20, 0.80, 0.21])
    treatment = np.array([1, 1, 0])

    result = nearest_neighbour_match(
        propensity,
        treatment,
        replacement=False,
        caliper=0.05,
    )

    assert result.n_matched == 1
    assert result.n_unmatched_treated == 1
    np.testing.assert_array_equal(
        result.unmatched_treated_indices,
        [1],
    )


def test_matching_rejects_invalid_caliper():
    propensity = np.array([0.20, 0.30, 0.25, 0.35])
    treatment = np.array([1, 1, 0, 0])

    with pytest.raises(ValueError, match="Caliper"):
        nearest_neighbour_match(
            propensity,
            treatment,
            caliper=0,
        )


def test_matching_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        nearest_neighbour_match(
            np.array([0.2, 0.4]),
            np.array([0, 1, 0]),
        )


def test_matching_rejects_invalid_treatment():
    with pytest.raises(ValueError, match="Treatment must contain"):
        nearest_neighbour_match(
            np.array([0.2, 0.4, 0.6]),
            np.array([0, 1, 2]),
        )


def test_matching_rejects_missing_group_within_support():
    with pytest.raises(ValueError, match="within common support"):
        nearest_neighbour_match(
            np.array([0.2, 0.4]),
            np.array([0, 1]),
            in_support=np.array([True, False]),
        )

def test_smd_is_zero_for_identical_groups():
    treated = pd.DataFrame({"age": [20, 30, 40]})
    control = pd.DataFrame({"age": [20, 30, 40]})

    result = standardized_mean_differences(
        treated, control, ["age"]
    )

    assert result["age"] == 0.0


def test_smd_detects_covariate_imbalance():
    treated = pd.DataFrame({"age": [40, 50, 60]})
    control = pd.DataFrame({"age": [20, 25, 30]})

    result = standardized_mean_differences(
        treated, control, ["age"]
    )

    assert result["age"] > 0.10


def test_smd_handles_zero_variance():
    treated = pd.DataFrame({"segment": [1, 1, 1]})
    control = pd.DataFrame({"segment": [1, 1, 1]})

    result = standardized_mean_differences(
        treated, control, ["segment"]
    )

    assert result["segment"] == 0.0


def test_smd_rejects_missing_covariates():
    treated = pd.DataFrame({"age": [20, 30, 40]})
    control = pd.DataFrame({"income": [100, 200, 300]})

    with pytest.raises(ValueError, match="Missing covariates"):
        standardized_mean_differences(
            treated, control, ["age"]
        )

def test_matched_sample_report_calculates_att():
    df = pd.DataFrame({
        "age": [30, 50, 29, 49],
        "treatment": [1, 1, 0, 0],
        "outcome": [1, 0, 0, 0],
    })

    propensity = np.array([0.20, 0.80, 0.21, 0.79])

    matches = nearest_neighbour_match(
        propensity,
        df["treatment"].to_numpy(),
        replacement=False,
    )

    report = evaluate_matched_sample(
        df,
        matches,
        covariates=["age"],
    )

    assert report.att == 0.5
    assert report.n_matched == 2
    assert report.n_unmatched_treated == 0


def test_matched_sample_report_measures_balance():
    df = pd.DataFrame({
        "age": [40, 60, 20, 40, 60],
        "treatment": [1, 1, 0, 0, 0],
        "outcome": [1, 1, 0, 0, 0],
    })

    propensity = np.array([0.20, 0.80, 0.10, 0.21, 0.79])

    matches = nearest_neighbour_match(
        propensity,
        df["treatment"].to_numpy(),
        replacement=False,
    )

    report = evaluate_matched_sample(
        df,
        matches,
        covariates=["age"],
    )

    assert abs(report.smd_after["age"]) < abs(report.smd_before["age"])
    assert report.balance_improved["age"] is True


def test_matched_sample_report_calculates_mean_distance():
    df = pd.DataFrame({
        "age": [30, 50, 29, 49],
        "treatment": [1, 1, 0, 0],
        "outcome": [1, 0, 0, 0],
    })

    propensity = np.array([0.20, 0.80, 0.21, 0.79])

    matches = nearest_neighbour_match(
        propensity,
        df["treatment"].to_numpy(),
        replacement=False,
    )

    report = evaluate_matched_sample(
        df,
        matches,
        covariates=["age"],
    )

    assert np.isclose(report.mean_matching_distance, 0.01)


def test_matched_sample_report_rejects_invalid_indices():
    df = pd.DataFrame({
        "age": [30, 50, 29, 49],
        "treatment": [1, 1, 0, 0],
        "outcome": [1, 0, 0, 0],
    })

    matches = MatchResult(
        treated_indices=np.array([0]),
        control_indices=np.array([10]),
        distances=np.array([0.01]),
        unmatched_treated_indices=np.array([1]),
        n_matched=1,
        n_unmatched_treated=1,
        n_unique_controls_used=1,
        replacement=False,
        caliper=None,
    )

    with pytest.raises(ValueError, match="outside the dataset"):
        evaluate_matched_sample(
            df,
            matches,
            covariates=["age"],
        )

def test_matched_sample_report_rejects_missing_treatment_column():
    df = pd.DataFrame({
        "age": [30, 50, 29, 49],
        "outcome": [1, 0, 0, 0],
    })

    matches = MatchResult(
        treated_indices=np.array([0]),
        control_indices=np.array([2]),
        distances=np.array([0.01]),
        unmatched_treated_indices=np.array([1]),
        n_matched=1,
        n_unmatched_treated=1,
        n_unique_controls_used=1,
        replacement=False,
        caliper=None,
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_matched_sample(
            df,
            matches,
            covariates=["age"],
        )

@pytest.mark.parametrize(
    "distances, error_message",
    [
        (np.array([]), "Match distances do not match n_matched"),
        (np.array([np.nan]), "finite non-negative values"),
        (np.array([np.inf]), "finite non-negative values"),
        (np.array([-0.01]), "finite non-negative values"),
    ],
)
def test_matched_sample_report_rejects_invalid_distances(
    distances, error_message
):
    df = pd.DataFrame({
        "treatment": [1, 0],
        "age": [30, 29],
        "outcome": [1, 0],
    })

    matches = MatchResult(
        treated_indices=np.array([0]),
        control_indices=np.array([1]),
        distances=distances,
        unmatched_treated_indices=np.array([], dtype=int),
        n_matched=1,
        n_unmatched_treated=0,
        n_unique_controls_used=1,
        replacement=False,
        caliper=None,
    )

    with pytest.raises(ValueError, match=error_message):
        evaluate_matched_sample(
            df,
            matches,
            covariates=["age"],
        )        