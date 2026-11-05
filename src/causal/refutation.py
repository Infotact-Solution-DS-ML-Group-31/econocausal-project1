"""Causal refutation suite implementing DoWhy validation tests."""

from typing import Any, Dict, Optional
import numpy as np


def run_causal_refutations(
    causal_model: Any,
    causal_estimate: Any,
    num_simulations: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute standard DoWhy refutation tests to validate causal identification.

    Tests performed:
      1. Placebo Treatment Refuter:
         Replaces treatment with random noise. The estimated causal effect should drop to ~0.
      2. Random Common Cause Refuter:
         Adds an unobserved random common cause. The estimated causal effect should remain stable.
      3. Data Subset Refuter:
         Re-estimates causal effect on 80% bootstrap sub-samples. The effect should remain robust.

    Parameters
    ----------
    causal_model : dowhy.CausalModel
        The instantiated causal model with dataset and DAG specification.
    causal_estimate : dowhy.CausalEstimate
        The estimated causal effect object produced by causal_model.estimate_effect().
    num_simulations : int, default=5
        Number of simulation / permutation runs per refuter (kept fast for CI/testing).
    random_seed : int, default=42
        Seed for stochastic refutations.

    Returns
    -------
    Dict[str, Any]
        Dictionary detailing individual test results, p-values, and mathematical pass/fail status.
    """
    orig_effect = float(causal_estimate.value)
    target_estimand = getattr(
        causal_estimate,
        "target_estimand",
        getattr(causal_model, "identified_estimand", None),
    )
    if target_estimand is None:
        target_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)

    results: Dict[str, Any] = {
        "original_effect": orig_effect,
        "refutations": {},
        "all_passed": True,
    }

    # Ensure RandomState instance for DoWhy refuters that call rng.normal(...)
    rng_state = (
        np.random.RandomState(random_seed)
        if isinstance(random_seed, int)
        else random_seed
    )

    # 1. Placebo Treatment Refuter
    try:
        ref_placebo = causal_model.refute_estimate(
            target_estimand,
            causal_estimate,
            method_name="placebo_treatment_refuter",
            num_simulations=num_simulations,
            random_state=random_seed,
        )
        placebo_effect = float(ref_placebo.new_effect)
        placebo_p_val = getattr(ref_placebo, "p_value", None)
        # Passed if effect collapsed towards zero (absolute magnitude < 0.05 or < 30% of original)
        placebo_passed = (
            abs(placebo_effect) < 0.05
            or abs(placebo_effect) < 0.30 * max(abs(orig_effect), 1e-4)
        )
        results["refutations"]["placebo_treatment"] = {
            "method": "placebo_treatment_refuter",
            "new_effect": placebo_effect,
            "p_value": float(placebo_p_val) if placebo_p_val is not None and not np.isnan(placebo_p_val) else None,
            "passed": bool(placebo_passed),
            "description": "Replaces treatment with random noise; effect should drop to ~0.",
        }
        if not placebo_passed:
            results["all_passed"] = False
    except Exception as exc:
        results["refutations"]["placebo_treatment"] = {
            "error": str(exc),
            "passed": False,
        }
        results["all_passed"] = False

    # 2. Random Common Cause Refuter
    try:
        ref_random = causal_model.refute_estimate(
            target_estimand,
            causal_estimate,
            method_name="random_common_cause",
            num_simulations=num_simulations,
            random_state=rng_state,
        )
        random_effect = float(ref_random.new_effect)
        random_p_val = getattr(ref_random, "p_value", None)
        # Passed if the estimate remains stable after adding a random confounder (abs diff < 0.05 or rel diff < 35%)
        abs_diff = abs(random_effect - orig_effect)
        rel_diff = abs_diff / max(abs(orig_effect), 1e-4)
        random_passed = (abs_diff < 0.05) or (rel_diff < 0.35)
        results["refutations"]["random_common_cause"] = {
            "method": "random_common_cause",
            "new_effect": random_effect,
            "p_value": float(random_p_val) if random_p_val is not None and not np.isnan(random_p_val) else None,
            "passed": bool(random_passed),
            "absolute_diff": float(abs_diff),
            "relative_drift": float(rel_diff),
            "description": "Adds random unobserved confounder; effect should remain stable.",
        }
        if not random_passed:
            results["all_passed"] = False
    except Exception as exc:
        results["refutations"]["random_common_cause"] = {
            "error": str(exc),
            "passed": False,
        }
        results["all_passed"] = False

    # 3. Data Subset Refuter (80% bootstrap subsets)
    try:
        ref_subset = causal_model.refute_estimate(
            target_estimand,
            causal_estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
            num_simulations=num_simulations,
            random_state=random_seed,
        )
        subset_effect = float(ref_subset.new_effect)
        subset_p_val = getattr(ref_subset, "p_value", None)
        # Passed if the effect on 80% random subsets remains stable (abs diff < 0.05 or rel diff < 35%)
        abs_diff_subset = abs(subset_effect - orig_effect)
        rel_diff_subset = abs_diff_subset / max(abs(orig_effect), 1e-4)
        subset_passed = (abs_diff_subset < 0.05) or (rel_diff_subset < 0.35)
        results["refutations"]["data_subset"] = {
            "method": "data_subset_refuter",
            "new_effect": subset_effect,
            "p_value": float(subset_p_val) if subset_p_val is not None and not np.isnan(subset_p_val) else None,
            "passed": bool(subset_passed),
            "absolute_diff": float(abs_diff_subset),
            "relative_drift": float(rel_diff_subset),
            "description": "Re-estimates on 80% bootstrap subsets; effect should remain robust.",
        }
        if not subset_passed:
            results["all_passed"] = False
    except Exception as exc:
        results["refutations"]["data_subset"] = {
            "error": str(exc),
            "passed": False,
        }
        results["all_passed"] = False

    return results
