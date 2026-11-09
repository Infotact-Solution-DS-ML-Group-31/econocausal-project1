
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class PropensityScoreResult:
    propensity_scores: np.ndarray
    n_observations: int
    n_treated: int
    n_control: int
    model: LogisticRegression


def estimate_propensity_scores(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "treatment",
    max_iter: int = 2000,
    random_state: int = 42,
) -> PropensityScoreResult:
    """
    Estimate propensity scores using logistic regression.

    Propensity score:
        P(Treatment = 1 | pre-treatment covariates)

    Covariates must be numeric, complete, finite, and measured
    before treatment assignment. This function does not establish
    causal identification or adequate common support.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if not covariates:
        raise ValueError("At least one pre-treatment covariate is required.")

    if len(covariates) != len(set(covariates)):
        raise ValueError("Covariates must not contain duplicates.")

    if treatment_col in covariates:
        raise ValueError("Treatment cannot be included as a covariate.")

    required = set(covariates) | {treatment_col}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[list(covariates) + [treatment_col]].copy()

    if data.isna().any().any():
        raise ValueError("Missing values are not allowed.")

    try:
        data = data.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All covariates and treatment must be numeric.") from exc

    values = data.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("All input values must be finite.")

    treatment_raw = data[treatment_col].to_numpy(dtype=float)

    if not np.isin(treatment_raw, [0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    treatment = treatment_raw.astype(int)
    n_treated = int(treatment.sum())
    n_control = len(treatment) - n_treated

    if n_treated == 0 or n_control == 0:
        raise ValueError("Both treatment groups are required.")


    features = data[covariates].to_numpy(dtype=float)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(features, treatment)

    propensity_scores = model.predict_proba(features)[:, 1]

    if not np.isfinite(propensity_scores).all():
        raise ValueError("Propensity estimation produced non-finite values.")

    return PropensityScoreResult(
        propensity_scores=propensity_scores,
        n_observations=len(data),
        n_treated=n_treated,
        n_control=n_control,
        model=model,
    )

@dataclass
class CommonSupportResult:
    lower_bound: float
    upper_bound: float
    in_support: np.ndarray
    n_in_support: int
    n_outside_support: int
    n_treated_in_support: int
    n_control_in_support: int


def check_common_support(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
) -> CommonSupportResult:
    """
    Identify observations within the overlapping propensity-score range.

    The common-support interval is the intersection of the observed
    treated and control propensity-score ranges.
    """

    propensity_scores = np.asarray(propensity_scores, dtype=float)
    treatment = np.asarray(treatment)

    if propensity_scores.ndim != 1 or treatment.ndim != 1:
        raise ValueError("Propensity scores and treatment must be one-dimensional.")

    if len(propensity_scores) != len(treatment):
        raise ValueError("Propensity scores and treatment must have equal lengths.")

    if len(propensity_scores) == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.isfinite(propensity_scores).all():
        raise ValueError("Propensity scores must be finite.")

    if ((propensity_scores <= 0) | (propensity_scores >= 1)).any():
        raise ValueError("Propensity scores must be strictly between 0 and 1.")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    treatment = treatment.astype(int)

    treated_scores = propensity_scores[treatment == 1]
    control_scores = propensity_scores[treatment == 0]

    if len(treated_scores) == 0 or len(control_scores) == 0:
        raise ValueError("Both treatment groups are required.")

    lower_bound = max(
        float(treated_scores.min()),
        float(control_scores.min()),
    )
    upper_bound = min(
        float(treated_scores.max()),
        float(control_scores.max()),
    )

    if lower_bound > upper_bound:
        raise ValueError("No common support exists between treatment groups.")

    in_support = (
        (propensity_scores >= lower_bound)
        & (propensity_scores <= upper_bound)
    )

    return CommonSupportResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        in_support=in_support,
        n_in_support=int(in_support.sum()),
        n_outside_support=int((~in_support).sum()),
        n_treated_in_support=int((in_support & (treatment == 1)).sum()),
        n_control_in_support=int((in_support & (treatment == 0)).sum()),
    )

@dataclass
class MatchResult:
    treated_indices: np.ndarray
    control_indices: np.ndarray
    distances: np.ndarray
    unmatched_treated_indices: np.ndarray
    n_matched: int
    n_unmatched_treated: int
    n_unique_controls_used: int
    replacement: bool
    caliper: float | None


def nearest_neighbour_match(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    caliper: float | None = None,
    replacement: bool = False,
    in_support: np.ndarray | None = None,
) -> MatchResult:
    """
    Perform greedy 1:1 nearest-neighbour propensity-score matching.

    Treated observations are processed in ascending propensity-score
    order. With replacement, a control can be matched more than once.
    Without replacement, each control can be used only once.

    Caliper is the maximum permitted absolute propensity-score
    difference. An optional common-support mask restricts eligibility.
    """

    propensity_scores = np.asarray(propensity_scores, dtype=float)
    treatment = np.asarray(treatment)

    if propensity_scores.ndim != 1 or treatment.ndim != 1:
        raise ValueError("Propensity scores and treatment must be one-dimensional.")

    if len(propensity_scores) != len(treatment):
        raise ValueError("Propensity scores and treatment must have equal lengths.")

    if len(propensity_scores) == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.isfinite(propensity_scores).all():
        raise ValueError("Propensity scores must be finite.")

    if ((propensity_scores <= 0) | (propensity_scores >= 1)).any():
        raise ValueError("Propensity scores must be strictly between 0 and 1.")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    treatment = treatment.astype(int)

    if caliper is not None:
        if not np.isfinite(caliper) or caliper <= 0:
            raise ValueError("Caliper must be a positive finite number.")

    if not isinstance(replacement, (bool, np.bool_)):
        raise ValueError("replacement must be a boolean.")

    if in_support is None:
        support_mask = np.ones(len(treatment), dtype=bool)
    else:
        support_mask = np.asarray(in_support)

        if support_mask.ndim != 1 or len(support_mask) != len(treatment):
            raise ValueError("Common-support mask must match input length.")

        if not np.isin(support_mask, [True, False]).all():
            raise ValueError("Common-support mask must contain boolean values.")

        support_mask = support_mask.astype(bool)

    eligible_treated = np.flatnonzero(support_mask & (treatment == 1))
    eligible_controls = np.flatnonzero(support_mask & (treatment == 0))

    if len(eligible_treated) == 0 or len(eligible_controls) == 0:
        raise ValueError(
            "Both treatment groups must have observations within common support."
        )

    # Process treated observations in ascending propensity-score order.
    order = np.argsort(
        propensity_scores[eligible_treated],
        kind="stable",
    )
    treated_order = eligible_treated[order]

    available_controls = eligible_controls.copy()

    matched_treated = []
    matched_controls = []
    distances = []
    unmatched_treated = []

    for treated_idx in treated_order:
        if len(available_controls) == 0:
            unmatched_treated.append(int(treated_idx))
            continue

        candidate_distances = np.abs(
            propensity_scores[available_controls]
            - propensity_scores[treated_idx]
        )

        nearest_position = int(np.argmin(candidate_distances))
        nearest_distance = float(candidate_distances[nearest_position])

        if caliper is not None and nearest_distance > caliper:
            unmatched_treated.append(int(treated_idx))
            continue

        control_idx = int(available_controls[nearest_position])

        matched_treated.append(int(treated_idx))
        matched_controls.append(control_idx)
        distances.append(nearest_distance)

        if not replacement:
            available_controls = np.delete(
                available_controls,
                nearest_position,
            )

    treated_indices = np.asarray(matched_treated, dtype=int)
    control_indices = np.asarray(matched_controls, dtype=int)
    distance_array = np.asarray(distances, dtype=float)
    unmatched_array = np.asarray(unmatched_treated, dtype=int)

    return MatchResult(
        treated_indices=treated_indices,
        control_indices=control_indices,
        distances=distance_array,
        unmatched_treated_indices=unmatched_array,
        n_matched=len(treated_indices),
        n_unmatched_treated=len(unmatched_array),
        n_unique_controls_used=len(np.unique(control_indices)),
        replacement=bool(replacement),
        caliper=caliper,
    )

def standardized_mean_differences(
    treated: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
) -> dict[str, float]:
    """
    Calculate standardized mean differences between treated and
    control groups for the specified numeric covariates.

    Positive values indicate a higher treated-group mean.
    Absolute values closer to zero indicate better balance.
    """

    if not isinstance(treated, pd.DataFrame) or not isinstance(control, pd.DataFrame):
        raise TypeError("treated and control must be pandas DataFrames.")

    if treated.empty or control.empty:
        raise ValueError("Both groups must contain observations.")

    if not covariates:
        raise ValueError("At least one covariate is required.")

    if len(covariates) != len(set(covariates)):
        raise ValueError("Covariates must not contain duplicates.")

    missing = sorted(
        set(covariates) - set(treated.columns)
        | (set(covariates) - set(control.columns))
    )

    if missing:
        raise ValueError(f"Missing covariates: {missing}")

    results = {}

    for column in covariates:
        try:
            treated_values = pd.to_numeric(
                treated[column], errors="raise"
            ).to_numpy(dtype=float)

            control_values = pd.to_numeric(
                control[column], errors="raise"
            ).to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Covariate '{column}' must be numeric."
            ) from exc

        if not np.isfinite(treated_values).all() or not np.isfinite(control_values).all():
            raise ValueError(
                f"Covariate '{column}' contains missing or non-finite values."
            )

        if len(treated_values) < 2 or len(control_values) < 2:
            raise ValueError(
                f"At least two observations per group are required for '{column}'."
            )

        treated_mean = float(np.mean(treated_values))
        control_mean = float(np.mean(control_values))

        treated_variance = float(np.var(treated_values, ddof=1))
        control_variance = float(np.var(control_values, ddof=1))

        pooled_sd = float(
            np.sqrt((treated_variance + control_variance) / 2)
        )

        mean_difference = treated_mean - control_mean

        if pooled_sd == 0:
            results[column] = (
                0.0 if mean_difference == 0 else float("inf")
            )
        else:
            results[column] = mean_difference / pooled_sd

    return results

@dataclass
class MatchedSampleReport:
    att: float
    n_matched: int
    n_unmatched_treated: int
    mean_matching_distance: float
    smd_before: dict[str, float]
    smd_after: dict[str, float]
    balance_improved: dict[str, bool]


def evaluate_matched_sample(
    df: pd.DataFrame,
    match_result: MatchResult,
    covariates: list[str],
    outcome_col: str = "outcome",
) -> MatchedSampleReport:
    """
    Evaluate matched outcomes and covariate balance.

    ATT is calculated as the mean treated outcome minus the mean
    matched-control outcome across matched pairs.

    Matching indices are positional indices, as returned by
    nearest_neighbour_match().
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if not isinstance(match_result, MatchResult):
        raise TypeError("match_result must be a MatchResult.")

    if match_result.n_matched == 0:
        raise ValueError("At least one matched pair is required.")

    required = set(covariates) | {outcome_col, "treatment"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    treated_indices = np.asarray(
        match_result.treated_indices, dtype=int
    )
    control_indices = np.asarray(
        match_result.control_indices, dtype=int
    )

    if (
        len(treated_indices) != match_result.n_matched
        or len(control_indices) != match_result.n_matched
    ):
        raise ValueError("Matched indices do not match n_matched.")

    if (
        (treated_indices < 0).any()
        or (control_indices < 0).any()
        or (treated_indices >= len(df)).any()
        or (control_indices >= len(df)).any()
    ):
        raise ValueError("Matched indices are outside the dataset.")

    distances = np.asarray(match_result.distances, dtype=float)

    if len(distances) != match_result.n_matched:
        raise ValueError("Match distances do not match n_matched.")

    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError(
            "Match distances must contain only finite non-negative values."
        )

    treatment = pd.to_numeric(
        df["treatment"], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(treatment).all() or not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must contain only finite 0 and 1 values.")

    if not (treatment[treated_indices] == 1).all():
        raise ValueError("Treated indices must refer to treated observations.")

    if not (treatment[control_indices] == 0).all():
        raise ValueError("Control indices must refer to control observations.")

    outcome = pd.to_numeric(
        df[outcome_col], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(outcome).all():
        raise ValueError("Outcome must contain only finite numeric values.")

    treated_matched = df.iloc[treated_indices]
    control_matched = df.iloc[control_indices]

    treated_outcomes = outcome[treated_indices]
    control_outcomes = outcome[control_indices]

    att = float(np.mean(treated_outcomes - control_outcomes))

    original_treated = df.loc[treatment == 1]
    original_control = df.loc[treatment == 0]

    smd_before = standardized_mean_differences(
        original_treated,
        original_control,
        covariates,
    )

    smd_after = standardized_mean_differences(
        treated_matched,
        control_matched,
        covariates,
    )

    balance_improved = {
        column: abs(smd_after[column]) < abs(smd_before[column])
        for column in covariates
    }

    mean_distance = float(np.mean(distances))
    return MatchedSampleReport(
        att=att,
        n_matched=match_result.n_matched,
        n_unmatched_treated=match_result.n_unmatched_treated,
        mean_matching_distance=mean_distance,
        smd_before=smd_before,
        smd_after=smd_after,
        balance_improved=balance_improved,
    )

