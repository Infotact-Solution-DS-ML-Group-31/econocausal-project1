"""Propensity score estimation, common support evaluation, and caliper matching."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PropensityScoreMatcher:
    """Estimates propensity scores and performs nearest-neighbor caliper matching.

    Assesses balance via Standardized Mean Differences (SMD) before and after matching.
    """

    def __init__(
        self,
        caliper: Optional[float] = 0.05,
        random_state: int = 42,
    ):
        """Initialize PropensityScoreMatcher.

        Parameters
        ----------
        caliper : float, optional
            Maximum allowable difference in propensity scores for a match.
            If None, defaults to 0.2 * standard deviation of logit propensity scores.
        random_state : int, default=42
            Random seed for logistic regression model.
        """
        self.caliper = caliper
        self.random_state = random_state
        self.model: Optional[LogisticRegression] = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    @staticmethod
    def compute_smd(
        treated: np.ndarray,
        control: np.ndarray,
    ) -> float:
        """Compute Standardized Mean Difference (SMD) between treated and control groups.

        SMD = |mean(T) - mean(C)| / sqrt((var(T) + var(C)) / 2)
        """
        if len(treated) == 0 or len(control) == 0:
            return 0.0

        mean_t = np.mean(treated)
        mean_c = np.mean(control)
        var_t = np.var(treated, ddof=1) if len(treated) > 1 else 0.0
        var_c = np.var(control, ddof=1) if len(control) > 1 else 0.0

        pooled_std = np.sqrt((var_t + var_c) / 2.0)
        if pooled_std < 1e-8 or np.isnan(pooled_std):
            return 0.0

        return float(np.abs(mean_t - mean_c) / pooled_std)

    def fit_predict_propensity(
        self,
        X: pd.DataFrame,
        T: pd.Series,
    ) -> pd.Series:
        """Train logistic regression model predicting P(T > 0 | X).

        Parameters
        ----------
        X : pd.DataFrame
            Covariates matrix.
        T : pd.Series
            Treatment vector (where T > 0 indicates receipt of any discount).

        Returns
        -------
        pd.Series
            Estimated propensity score e(X) for each customer.
        """
        self.feature_names_ = list(X.columns)
        binary_treatment = (T > 0).astype(int).to_numpy()

        # If all treated or all control, return naive propensity
        if len(np.unique(binary_treatment)) < 2:
            return pd.Series(0.5, index=X.index)

        self.model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.model.fit(X.to_numpy(dtype=float), binary_treatment)
        self.is_fitted_ = True

        # Predict probability of treatment P(D=1 | X)
        propensity_scores = self.model.predict_proba(X.to_numpy(dtype=float))[:, 1]
        # Clip to ensure numerical stability in positivity assumption (0 < e(X) < 1)
        propensity_scores = np.clip(propensity_scores, 1e-5, 1.0 - 1e-5)

        return pd.Series(propensity_scores, index=X.index, name="propensity_score")

    def evaluate_common_support(
        self,
        propensity_scores: pd.Series,
        T: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluate the Positivity / Common Support assumption across groups.

        Parameters
        ----------
        propensity_scores : pd.Series
            Propensity scores.
        T : pd.Series
            Treatment vector.

        Returns
        -------
        Dict[str, Any]
            Common support bounds and coverage statistics.
        """
        binary_treatment = (T > 0).astype(int)
        treated_scores = propensity_scores[binary_treatment == 1].to_numpy()
        control_scores = propensity_scores[binary_treatment == 0].to_numpy()

        if len(treated_scores) == 0 or len(control_scores) == 0:
            return {"has_common_support": False, "overlap_fraction": 0.0}

        min_t, max_t = float(np.min(treated_scores)), float(np.max(treated_scores))
        min_c, max_c = float(np.min(control_scores)), float(np.max(control_scores))

        cs_min = max(min_t, min_c)
        cs_max = min(max_t, max_c)
        has_support = cs_max > cs_min

        # Compute fraction of units within the common support window
        in_support = (propensity_scores >= cs_min) & (propensity_scores <= cs_max)
        overlap_frac = float(in_support.mean())

        return {
            "has_common_support": bool(has_support),
            "common_support_min": float(cs_min),
            "common_support_max": float(cs_max),
            "treated_min": min_t,
            "treated_max": max_t,
            "control_min": min_c,
            "control_max": max_c,
            "overlap_fraction": overlap_frac,
            "units_in_support_count": int(in_support.sum()),
        }

    def match(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform 1-to-1 nearest neighbor matching within caliper without replacement.

        Parameters
        ----------
        X : pd.DataFrame
            Covariates matrix.
        T : pd.Series
            Treatment vector.
        df : pd.DataFrame, optional
            Original customer dataset to index matched rows from. If None, uses X.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            (matched_dataframe, balance_report)
        """
        propensity_scores = self.fit_predict_propensity(X, T)
        common_support_info = self.evaluate_common_support(propensity_scores, T)

        binary_t = (T > 0).astype(int)
        treated_idx = X.index[binary_t == 1].to_numpy()
        control_idx = X.index[binary_t == 0].to_numpy()

        # Determine caliper width
        caliper_width = self.caliper
        if caliper_width is None:
            # Standard Rosenbaum & Rubin recommendation: 0.2 * std of logit propensity
            ps_clipped = np.clip(propensity_scores.to_numpy(), 0.01, 0.99)
            logit_ps = np.log(ps_clipped / (1.0 - ps_clipped))
            caliper_width = float(0.20 * np.std(logit_ps))
            caliper_width = max(caliper_width, 0.02)

        matched_treated_idx: List[Any] = []
        matched_control_idx: List[Any] = []
        available_controls = set(control_idx)

        # Sort treated units by propensity score to match extreme units first
        treated_sorted = treated_idx[np.argsort(propensity_scores.loc[treated_idx].to_numpy())]

        for t_i in treated_sorted:
            if not available_controls:
                break
            score_t = propensity_scores.loc[t_i]
            avail_list = np.array(list(available_controls))
            diffs = np.abs(propensity_scores.loc[avail_list].to_numpy() - score_t)
            best_idx = np.argmin(diffs)

            if diffs[best_idx] <= caliper_width:
                matched_c = avail_list[best_idx]
                matched_treated_idx.append(t_i)
                matched_control_idx.append(matched_c)
                available_controls.remove(matched_c)

        # Assemble matched DataFrame
        matched_indices = matched_treated_idx + matched_control_idx
        source_df = df if df is not None else X.copy()
        matched_df = source_df.loc[matched_indices].copy()
        matched_df["propensity_score"] = propensity_scores.loc[matched_indices]

        # Compute Pre-Match and Post-Match SMD across all covariates
        pre_match_smd: Dict[str, float] = {}
        post_match_smd: Dict[str, float] = {}

        for col in X.columns:
            # Pre-match
            smd_pre = self.compute_smd(
                X.loc[treated_idx, col].to_numpy(dtype=float),
                X.loc[control_idx, col].to_numpy(dtype=float),
            )
            pre_match_smd[col] = float(smd_pre)

            # Post-match
            if matched_treated_idx and matched_control_idx:
                smd_post = self.compute_smd(
                    X.loc[matched_treated_idx, col].to_numpy(dtype=float),
                    X.loc[matched_control_idx, col].to_numpy(dtype=float),
                )
            else:
                smd_post = smd_pre
            post_match_smd[col] = float(smd_post)

        mean_pre_smd = float(np.mean(list(pre_match_smd.values())))
        mean_post_smd = float(np.mean(list(post_match_smd.values())))
        max_post_smd = float(np.max(list(post_match_smd.values()))) if post_match_smd else 0.0

        # Balance criteria: mean post-match SMD < 0.10 or significant reduction (> 50%)
        is_balanced = bool(mean_post_smd < 0.10 or mean_post_smd < 0.50 * max(mean_pre_smd, 0.01))

        balance_report = {
            "caliper": caliper_width,
            "n_matched_pairs": len(matched_treated_idx),
            "common_support": common_support_info,
            "pre_match_smd": pre_match_smd,
            "post_match_smd": post_match_smd,
            "mean_pre_match_smd": mean_pre_smd,
            "mean_post_match_smd": mean_post_smd,
            "max_post_match_smd": max_post_smd,
            "is_balanced": is_balanced,
        }

        return matched_df, balance_report
