"""Causal Inference, Uplift Modeling, and Refutation Engine for EconoCausal."""

# Compatibility shims for NetworkX 3+ and pandas 3+ with DoWhy 0.8
try:
    import networkx as nx

    if not hasattr(nx.algorithms, "d_separated"):
        import networkx.algorithms.d_separation as ds

        nx.algorithms.d_separated = ds.is_d_separator

    from dowhy.causal_estimator import CausalEstimate
    from dowhy.causal_estimators.regression_estimator import RegressionEstimator

    def _patched_estimate_effect(self, data_df=None, need_conditional_estimates=None):
        if data_df is None:
            data_df = self._data
        if need_conditional_estimates is None:
            need_conditional_estimates = self.need_conditional_estimates
        if not self.model:
            _, self.model = self._build_model()
        effect_estimate = self._do(self._treatment_value, data_df) - self._do(
            self._control_value, data_df
        )
        conditional_effect_estimates = None
        if need_conditional_estimates:
            conditional_effect_estimates = self._estimate_conditional_effects(
                self._estimate_effect_fn,
                effect_modifier_names=self._effect_modifier_names,
            )
        intercept_parameter = (
            self.model.params.iloc[0]
            if hasattr(self.model.params, "iloc")
            else self.model.params[0]
        )
        return CausalEstimate(
            estimate=effect_estimate,
            control_value=self._control_value,
            treatment_value=self._treatment_value,
            conditional_estimates=conditional_effect_estimates,
            target_estimand=self._target_estimand,
            realized_estimand_expr=self.symbolic_estimator,
            intercept=intercept_parameter,
        )

    RegressionEstimator._estimate_effect = _patched_estimate_effect
except Exception:
    pass

from src.causal.dataset import generate_observational_retail_data
from src.causal.estimator import DoubleMLEstimator
from src.causal.graph import build_causal_dag, identify_causal_effect
from src.causal.refutation import run_causal_refutations

__all__ = [
    "generate_observational_retail_data",
    "build_causal_dag",
    "identify_causal_effect",
    "DoubleMLEstimator",
    "run_causal_refutations",
]

