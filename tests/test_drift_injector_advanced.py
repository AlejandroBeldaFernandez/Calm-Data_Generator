"""
Tests for DriftInjector - Advanced Categorical Drift Methods
"""

import unittest
import pandas as pd
import numpy as np
from calm_data_generator.generators.drift import DriftInjector


class TestDriftInjectorCategorical(unittest.TestCase):
    """Tests for the new categorical drift injection methods."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "category": np.random.choice(["A", "B", "C"], n),
                "city": np.random.choice(["Madrid", "Barcelona", "Valencia"], n),
                "is_active": np.random.choice([True, False], n),
                "flag": np.random.choice([0, 1], n),
                "value": np.random.randn(n),
            }
        )
        self.injector = DriftInjector(auto_report=False, random_state=42)

    def test_categorical_frequency_drift_uniform(self):
        """Test that frequency drift changes category distribution."""
        original_counts = self.df["category"].value_counts(normalize=True)

        drifted_df = self.injector.inject_categorical_frequency_drift(
            df=self.df,
            feature_cols=["category"],
            drift_magnitude=0.5,
            perturbation="uniform",
        )

        drifted_counts = drifted_df["category"].value_counts(normalize=True)

        # Drifted distribution should be different
        self.assertFalse(original_counts.equals(drifted_counts))
        # Same shape
        self.assertEqual(len(self.df), len(drifted_df))

    def test_categorical_frequency_drift_invert(self):
        """Test invert perturbation strategy."""
        drifted_df = self.injector.inject_categorical_frequency_drift(
            df=self.df,
            feature_cols=["category"],
            drift_magnitude=1.0,
            perturbation="invert",
        )
        self.assertEqual(len(self.df), len(drifted_df))

    def test_typos_drift(self):
        """Test typo injection into string columns."""
        drifted_df = self.injector.inject_typos_drift(
            df=self.df,
            feature_cols=["city"],
            drift_magnitude=0.5,
            typo_density=1,
            typo_type="random",
        )

        # At least some values should have changed
        changed = (self.df["city"] != drifted_df["city"]).sum()
        self.assertGreater(changed, 0)

    def test_category_merge_drift(self):
        """Test merging categories."""
        drifted_df = self.injector.inject_category_merge_drift(
            df=self.df,
            col="category",
            categories_to_merge=["A", "B"],
            new_category_name="AB",
        )

        # A and B should be gone, AB should exist
        unique_vals = drifted_df["category"].unique()
        self.assertIn("AB", unique_vals)
        self.assertNotIn("A", unique_vals)
        self.assertNotIn("B", unique_vals)
        self.assertIn("C", unique_vals)

    def test_boolean_drift_true_false(self):
        """Test boolean flipping on True/False column."""
        original_true_count = self.df["is_active"].sum()

        drifted_df = self.injector.inject_boolean_drift(
            df=self.df,
            feature_cols=["is_active"],
            drift_magnitude=1.0,  # Flip all
        )

        drifted_true_count = drifted_df["is_active"].sum()

        # With magnitude 1.0, all values should be flipped
        # So original True count + drifted True count should equal total rows
        self.assertEqual(original_true_count + drifted_true_count, len(self.df))

    def test_boolean_drift_zero_one(self):
        """Test boolean flipping on 0/1 integer column."""
        original_ones = self.df["flag"].sum()

        drifted_df = self.injector.inject_boolean_drift(
            df=self.df,
            feature_cols=["flag"],
            drift_magnitude=1.0,
        )

        drifted_ones = drifted_df["flag"].sum()
        self.assertEqual(original_ones + drifted_ones, len(self.df))

    def test_conditional_drift_gradual(self):
        """Test that conditional drift works with gradual method."""
        drifted_df = self.injector.inject_conditional_drift(
            df=self.df,
            feature_cols=["value"],
            conditions=[{"column": "category", "operator": "==", "value": "A"}],
            drift_type="shift",
            drift_magnitude=0.5,
            drift_method="gradual",
        )

        # Only rows where category == 'A' should be affected
        a_rows_original = self.df[self.df["category"] == "A"]["value"]
        a_rows_drifted = drifted_df[self.df["category"] == "A"]["value"]

        # Values should be different
        self.assertFalse(a_rows_original.equals(a_rows_drifted))

        # Non-A rows should be unchanged
        non_a_original = self.df[self.df["category"] != "A"]["value"]
        non_a_drifted = drifted_df[self.df["category"] != "A"]["value"]
        pd.testing.assert_series_equal(non_a_original, non_a_drifted)


if __name__ == "__main__":
    unittest.main()
