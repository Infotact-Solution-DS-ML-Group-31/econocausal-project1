from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UpliftResult:
    auuc: float
    qini_coefficient: float
    random_auuc: float
    n_observations: int
    n_treated: int
    n_control: int
    curve: tuple[float, ...]
    random_curve: tuple[float, ...]
    warnings: list[str]


def evaluate_uplift(
    treatment: np.ndarray,
    outcome: np.ndarray,
    predicted_uplift: np.ndarray,
) -> UpliftResult:
    """
    Evaluate an uplift ranking using a Qini-style cumulative gain curve.

    Rows are ranked by predicted uplift in descending order.

    The curve measures cumulative treated outcomes minus the
    control outcome rate scaled by the treated count.

    Inputs must represent valid treatment assignment and outcomes.
    """

    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    predicted_uplift = np.asarray(predicted_uplift, dtype=float)

    if (
        treatment.ndim != 1
        or outcome.ndim != 1
        or predicted_uplift.ndim != 1
    ):
        raise ValueError("All inputs must be one-dimensional arrays.")

    if not (
        len(treatment) == len(outcome) == len(predicted_uplift)
    ):
        raise ValueError("All input arrays must have equal lengths.")

    if len(treatment) == 0:
        raise ValueError("Input arrays must not be empty.")

    if not (
        np.isfinite(treatment).all()
        and np.isfinite(outcome).all()
        and np.isfinite(predicted_uplift).all()
    ):
        raise ValueError("Inputs must contain only finite values.")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("Outcome must contain only 0 and 1.")

    n_treated = int(treatment.sum())
    n_control = len(treatment) - n_treated

    if n_treated == 0 or n_control == 0:
        raise ValueError("Both treatment groups are required.")

    treated_outcome_rate = float(
        outcome[treatment == 1].mean()
    )
    control_outcome_rate = float(
        outcome[treatment == 0].mean()
    )

    # Rank observations by predicted uplift.
    order = np.argsort(-predicted_uplift, kind="stable")

    ranked_treatment = treatment[order]
    ranked_outcome = outcome[order]

    cumulative_treated = np.cumsum(ranked_treatment)
    cumulative_control = np.cumsum(1 - ranked_treatment)

    cumulative_treated_outcomes = np.cumsum(
        ranked_outcome * ranked_treatment
    )
    cumulative_control_outcomes = np.cumsum(
        ranked_outcome * (1 - ranked_treatment)
    )

    # Cumulative Qini gain.
    curve = (
        cumulative_treated_outcomes
        - cumulative_control_outcomes * (n_treated / n_control)
    )

    # Random-targeting baseline, using overall group outcome rates.
    cumulative_random = (
        np.arange(1, len(treatment) + 1)
        / len(treatment)
    ) * (
        n_treated
        * (treated_outcome_rate - control_outcome_rate)
    )

    # Include the origin when calculating areas.
    curve_with_origin = np.concatenate(([0.0], curve))
    random_with_origin = np.concatenate(([0.0], cumulative_random))

    x = np.linspace(0.0, 1.0, len(treatment) + 1)

    auuc = float(np.trapezoid(curve_with_origin, x))
    random_auuc = float(np.trapezoid(random_with_origin, x))

    qini_coefficient = float(auuc - random_auuc)

    return UpliftResult(
        auuc=auuc,
        qini_coefficient=qini_coefficient,
        random_auuc=random_auuc,
        n_observations=len(treatment),
        n_treated=n_treated,
        n_control=n_control,
        curve=tuple(float(value) for value in curve_with_origin),
        random_curve=tuple(
            float(value) for value in random_with_origin
        ),
        warnings=[
            "Uplift evaluation requires valid treatment assignment and outcome data.",
            "The Qini curve is a ranking diagnostic, not proof of causal identification.",
            "Interpretation depends on the treatment-assignment mechanism and evaluation design.",
            "Do not interpret historical retail transaction prices as randomized treatments.",
        ],
    )