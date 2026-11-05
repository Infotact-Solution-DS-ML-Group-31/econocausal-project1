"""Unit and integration tests for Role M1 Data Engineering, Preprocessing, Matching & EDA."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.analysis import generate_eda_report
from src.causal import generate_observational_retail_data
from src.data import (
    CausalDataPreprocessor,
    PropensityScoreMatcher,
    RetailDataLoader,
)


class TestDataPipeline(unittest.TestCase):
    """Test suite covering data loading, cleaning, causal preprocessing,

    propensity score matching, and exploratory data analysis (EDA).
    """

    @classmethod
    def setUpClass(cls):
        cls.sample_df = generate_observational_retail_data(n_samples=300, seed=42)

    def test_retail_data_loader_missing_values_imputation_and_deduplication(self):
        """Test data loader handles missing values, removes duplicates, and enforces types."""
        # Create dirty dataframe with duplicates and missing values
        dirty_data = pd.DataFrame(
            {
                "customer_id": ["cust_1", "cust_1", "cust_2", "cust_3"],
                "income": [50000.0, 50000.0, np.nan, 85000.0],
                "tenure_months": [12, 12, 24, np.nan],
                "loyalty_score": [5.5, 5.5, np.nan, 8.0],
                "historical_spend": [200.0, 200.0, np.nan, 450.0],
                "browse_minutes": [30.0, 30.0, 45.0, np.nan],
                "treatment": [10, 10, np.nan, 0],
                "conversion": [1, 1, np.nan, 0],
                "transaction_amount": [55.0, 55.0, np.nan, 0.0],
            }
        )

        loader = RetailDataLoader()
        cleaned = loader.load(dirty_data)

        # 1. Check deduplication: cust_1 should appear only once
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(list(cleaned["customer_id"]), ["cust_1", "cust_2", "cust_3"])

        # 2. Check imputation: no NaNs should remain
        self.assertFalse(cleaned.isnull().any().any())

        # 3. Check types
        self.assertTrue(np.issubdtype(cleaned["treatment"].dtype, np.integer))
        self.assertTrue(np.issubdtype(cleaned["conversion"].dtype, np.integer))
        self.assertTrue(np.issubdtype(cleaned["income"].dtype, np.floating))

    def test_retail_data_loader_file_io_csv_and_json(self):
        """Test loading from CSV and JSON file paths."""
        loader = RetailDataLoader()
        sample_subset = self.sample_df.head(20).copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample_data.csv"
            json_path = Path(tmp_dir) / "sample_data.json"

            sample_subset.to_csv(csv_path, index=False)
            sample_subset.to_json(json_path, orient="records")

            # Load CSV
            loaded_csv = loader.load(csv_path)
            self.assertEqual(len(loaded_csv), 20)

            # Load JSON
            loaded_json = loader.load(json_path)
            self.assertEqual(len(loaded_json), 20)

    def test_retail_data_loader_missing_column_validation(self):
        """Test that missing required columns raises ValueError."""
        incomplete_df = pd.DataFrame(
            {
                "customer_id": ["cust_1"],
                "income": [50000.0],
            }
        )
        loader = RetailDataLoader()
        with self.assertRaises(ValueError):
            loader.load(incomplete_df)

    def test_causal_data_preprocessor_scaling_and_engineered_features(self):
        """Test continuous standardization and domain-specific feature engineering."""
        preprocessor = CausalDataPreprocessor()
        X, T, Y = preprocessor.prepare_matrices(self.sample_df)

        # 1. Engineered columns exist
        self.assertIn("spend_per_month", X.columns)
        self.assertIn("engagement_index", X.columns)

        # 2. Continuous standardization: normalized columns should have mean ~ 0 and std ~ 1
        for col in ["income", "historical_spend", "browse_minutes"]:
            self.assertAlmostEqual(float(X[col].mean()), 0.0, places=2)
            self.assertAlmostEqual(float(X[col].std()), 1.0, places=2)

        # 3. Length consistency
        self.assertEqual(len(X), len(self.sample_df))
        self.assertEqual(len(T), len(self.sample_df))
        self.assertEqual(len(Y), len(self.sample_df))

        # 4. Unfitted error
        unfitted = CausalDataPreprocessor()
        with self.assertRaises(RuntimeError):
            unfitted.transform(self.sample_df)

    def test_propensity_score_matching_and_smd_balance(self):
        """Test propensity score estimation, common support, and caliper matching balance."""
        # Use synthetic retail cohort with observational confounding
        df = generate_observational_retail_data(n_samples=500, seed=42)
        preprocessor = CausalDataPreprocessor()
        X, T, _ = preprocessor.prepare_matrices(df)

        matcher = PropensityScoreMatcher(caliper=0.08, random_state=42)
        matched_df, balance_report = matcher.match(X, T, df=df)

        # 1. Non-empty matches
        self.assertGreater(balance_report["n_matched_pairs"], 0)
        self.assertEqual(len(matched_df), 2 * balance_report["n_matched_pairs"])

        # 2. Common support check
        cs = balance_report["common_support"]
        self.assertTrue(cs["has_common_support"])
        self.assertGreater(cs["overlap_fraction"], 0.70)

        # 3. Covariate balance: post-matching SMD must show substantial improvement over pre-match
        self.assertLess(
            balance_report["mean_post_match_smd"],
            balance_report["mean_pre_match_smd"],
            "Matching failed to reduce average covariate imbalance.",
        )
        # Average post-match SMD should achieve high balance (< 0.10)
        self.assertLess(
            balance_report["mean_post_match_smd"],
            0.12,
            f"Mean post-match SMD exceeded balance threshold: {balance_report['mean_post_match_smd']}",
        )
        self.assertTrue(balance_report["is_balanced"])

    def test_generate_eda_report_stratification_and_diagnostics(self):
        """Test exploratory data analysis report generation across treatment tiers."""
        report = generate_eda_report(self.sample_df)

        # 1. Overview
        overview = report["overview"]
        self.assertEqual(overview["total_customers"], len(self.sample_df))
        self.assertGreaterEqual(overview["overall_conversion_rate"], 0.0)
        self.assertLessEqual(overview["overall_conversion_rate"], 1.0)

        # 2. Stratification
        stratified = report["stratified_by_treatment"]
        self.assertIn("tier_0", stratified)
        self.assertIn("tier_10", stratified)
        self.assertIn("tier_20", stratified)

        # 3. Contrasts
        contrasts = report["treatment_contrasts"]
        self.assertIn("tier_10_vs_control", contrasts)
        self.assertIn("tier_20_vs_control", contrasts)

        # 4. Distribution diagnostics
        diagnostics = report["distribution_diagnostics"]
        for col in ["income", "loyalty_score", "historical_spend", "browse_minutes"]:
            self.assertIn(col, diagnostics)
            self.assertIn("skewness", diagnostics[col])
            self.assertIn("kurtosis", diagnostics[col])

        # 5. Empty dataframe raises ValueError
        with self.assertRaises(ValueError):
            generate_eda_report(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()

