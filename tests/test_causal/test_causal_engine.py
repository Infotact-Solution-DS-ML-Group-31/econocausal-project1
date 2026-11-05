"""Unit and integration tests for Role M2 Causal ML, DAG identification, and Refutation engine."""

import unittest
import numpy as np
import pandas as pd

from src.api.schemas import CustomerDataPoint
from src.causal import (
    DoubleMLEstimator,
    build_causal_dag,
    generate_observational_retail_data,
    identify_causal_effect,
    run_causal_refutations,
)


class TestCausalEngine(unittest.TestCase):
    """Test suite covering observational retail data generation, Double ML CATE estimation,

    DoWhy DAG backdoor identification, and causal refutation suites.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = generate_observational_retail_data(n_samples=400, seed=42)

    def test_synthetic_dataset_generation_structure_and_distributions(self):
        """Test observational retail data synthesis asserting expected schema, values, and bounds."""
        n_samples = 250
        df = generate_observational_retail_data(n_samples=n_samples, seed=123)

        self.assertEqual(len(df), n_samples)
        expected_cols = [
            "customer_id",
            "income",
            "tenure_months",
            "loyalty_score",
            "historical_spend",
            "browse_minutes",
            "treatment",
            "conversion",
            "transaction_amount",
            "true_ite",
            "baseline_conversion_prob",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Check for non-null values
        self.assertFalse(df.isnull().any().any(), "Dataset contains unexpected null values.")

        # Check treatment assignment in {0, 10, 20}
        unique_treatments = set(df["treatment"].unique())
        self.assertTrue(
            unique_treatments.issubset({0, 10, 20}),
            f"Unexpected treatment levels: {unique_treatments}",
        )

        # Check binary conversions in {0, 1}
        self.assertTrue(set(df["conversion"].unique()).issubset({0, 1}))

        # Check baseline conversion probability in [0, 1]
        self.assertTrue((df["baseline_conversion_prob"] >= 0.0).all())
        self.assertTrue((df["baseline_conversion_prob"] <= 1.0).all())

        # Check non-negative transactions
        self.assertTrue((df["transaction_amount"] >= 0.0).all())
        # Customers who converted should have positive transaction spend
        converted_spends = df[df["conversion"] == 1]["transaction_amount"]
        if len(converted_spends) > 0:
            self.assertTrue((converted_spends > 0.0).all())

    def test_double_ml_training_and_ite_recovery(self):
        """Test Double ML estimator cross-fitting, ATE estimation, and positive ITE recovery on persuadables."""
        dml = DoubleMLEstimator(n_splits=3, random_state=42)
        dml.fit(self.df, treatment_col="treatment", outcome_col="conversion")

        self.assertTrue(dml.is_fitted_)
        self.assertIsInstance(dml.ate, float)

        # Predict ITEs
        predicted_ites = dml.predict_ite(self.df)
        self.assertEqual(len(predicted_ites), len(self.df))
        self.assertFalse(np.isnan(predicted_ites).any())

        # Test recovery: subset with high true ITE should have positive mean predicted ITE
        persuadables = self.df[self.df["true_ite_10"] > 0.10]
        if len(persuadables) > 10:
            pred_persuadables = dml.predict_ite(persuadables)
            # Uplift predictions on persuadables should be positive on average
            self.assertGreater(
                float(np.mean(pred_persuadables)),
                -0.01,
                "Predicted ITE on strong persuadables cohort was unexpectedly strongly negative.",
            )

        # Summary diagnostics
        summary = dml.summary()
        self.assertTrue(summary["is_fitted"])
        self.assertEqual(summary["n_features"], len(dml.feature_cols_))
        self.assertIn("tau_weights", summary)

    def test_predict_customer_datapoints_converts_to_pydantic_schema(self):
        """Test converting Causal ML DataFrame outputs into M3 CustomerDataPoint objects."""
        dml = DoubleMLEstimator(n_splits=3, random_state=42)
        dml.fit(self.df)

        sample_subset = self.df.head(15)
        datapoints = dml.predict_customer_datapoints(
            sample_subset, default_cost=10.0, default_value=60.0
        )

        self.assertEqual(len(datapoints), 15)
        for dp in datapoints:
            self.assertIsInstance(dp, CustomerDataPoint)
            self.assertTrue(dp.customer_id.startswith("cust_"))
            self.assertGreaterEqual(dp.baseline_conversion_prob, 0.0)
            self.assertLessEqual(dp.baseline_conversion_prob, 1.0)
            self.assertIsInstance(dp.predicted_ite, float)
            self.assertGreaterEqual(dp.treatment_cost, 0.0)
            self.assertGreater(dp.customer_value, 0.0)

    def test_causal_dag_specification_and_backdoor_identification(self):
        """Test build_causal_dag and DoWhy backdoor adjustment identification."""
        dag_str = build_causal_dag()
        self.assertIn("digraph", dag_str)
        self.assertIn("income -> treatment", dag_str)
        self.assertIn("treatment -> conversion", dag_str)

        # Identify backdoor estimand
        identified_estimand = identify_causal_effect(self.df)
        self.assertIsNotNone(identified_estimand)

        # Check backdoor adjustment set
        backdoors = identified_estimand.get_backdoor_variables()
        self.assertIn("income", backdoors)
        self.assertIn("loyalty_score", backdoors)
        self.assertIn("historical_spend", backdoors)
        self.assertIn("browse_minutes", backdoors)

        # Ensure attached causal model is valid
        self.assertIsNotNone(identified_estimand.causal_model)

    def test_refutation_suite_execution_and_stability(self):
        """Test executing Placebo, Random Common Cause, and Data Subset refutations."""
        identified_estimand = identify_causal_effect(self.df)
        model = identified_estimand.causal_model
        estimate = model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression"
        )
        self.assertIsNotNone(estimate.value)

        # Run refutation suite with 3 simulations for fast CI execution
        ref_results = run_causal_refutations(model, estimate, num_simulations=3, random_seed=42)

        self.assertIn("original_effect", ref_results)
        self.assertIn("refutations", ref_results)
        self.assertIn("all_passed", ref_results)

        ref_dict = ref_results["refutations"]
        self.assertIn("placebo_treatment", ref_dict)
        self.assertIn("random_common_cause", ref_dict)
        self.assertIn("data_subset", ref_dict)

        # Placebo treatment effect must be near zero
        placebo_new_effect = ref_dict["placebo_treatment"]["new_effect"]
        self.assertLess(
            abs(placebo_new_effect),
            0.15,
            f"Placebo treatment effect did not collapse toward 0: {placebo_new_effect}",
        )

        # Random common cause should not cause drastic divergence
        random_effect = ref_dict["random_common_cause"]["new_effect"]
        self.assertLess(
            abs(random_effect - estimate.value),
            0.10,
            f"Random common cause induced excessive drift: {random_effect} vs {estimate.value}",
        )

    def test_estimator_input_validation_and_errors(self):
        """Test input error handling for un-fitted estimators and missing columns."""
        unfitted_dml = DoubleMLEstimator()
        with self.assertRaises(RuntimeError):
            unfitted_dml.predict_ite(self.df)

        with self.assertRaises(RuntimeError):
            unfitted_dml.predict_customer_datapoints(self.df)

        with self.assertRaises(RuntimeError):
            _ = unfitted_dml.ate

        # Test fit with empty feature columns
        empty_df = pd.DataFrame({"treatment": [0, 10], "conversion": [0, 1]})
        with self.assertRaises(ValueError):
            unfitted_dml.fit(empty_df, feature_cols=[])


if __name__ == "__main__":
    unittest.main()

