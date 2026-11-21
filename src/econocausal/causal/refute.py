from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from econocausal.causal.estimate import estimate_ate_dml


@dataclass
class RefutationResult:
    observed_effect: float
    placebo_effects: tuple[float, ...]
    placebo_mean: float
    placebo_std: float
    fraction_at_least_as_extreme: float
    n_repeats: int
    method: str
    warnings: list[str]


def placebo_treatment_test(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    n_repeats: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
) -> RefutationResult:
    """
    Run a placebo-treatment falsification diagnostic.

    Treatment labels are shuffled while preserving the treated-group size.
    This diagnostic does not prove causal identification or eliminate
    unmeasured confounding.
    """

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    if treatment_col not in df.columns:
        raise ValueError(f"Missing treatment column: {treatment_col}")

    # Estimate using the actual treatment assignments.
    observed = estimate_ate_dml(
        df=df,
        covariates=covariates,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        n_splits=n_splits,
        random_state=random_state,
    )

    original_treatment = df[treatment_col].to_numpy(copy=True)
    rng = np.random.default_rng(random_state)

    placebo_effects = []

    for repeat in range(n_repeats):
        placebo_df = df.copy()
        placebo_df[treatment_col] = rng.permutation(original_treatment)

        placebo_estimate = estimate_ate_dml(
            df=placebo_df,
            covariates=covariates,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            n_splits=n_splits,
            random_state=random_state + repeat + 1,
        )

        placebo_effects.append(float(placebo_estimate.ate))

    placebo_array = np.asarray(placebo_effects, dtype=float)

    fraction_at_least_as_extreme = float(
        np.mean(np.abs(placebo_array) >= abs(observed.ate))
    )

    return RefutationResult(
        observed_effect=float(observed.ate),
        placebo_effects=tuple(placebo_effects),
        placebo_mean=float(np.mean(placebo_array)),
        placebo_std=float(np.std(placebo_array, ddof=1))
        if n_repeats > 1
        else 0.0,
        fraction_at_least_as_extreme=fraction_at_least_as_extreme,
        n_repeats=n_repeats,
        method="Randomized placebo-treatment test",
        warnings=[
            "A placebo test is a falsification diagnostic, not proof of causality.",
            "A small placebo effect does not rule out unmeasured confounding.",
            "Interpretation depends on the treatment-assignment mechanism.",
            "Do not apply this test to the historical retail dataset as if its observed prices were assigned treatments.",
        ],
    )