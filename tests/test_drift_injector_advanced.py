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


class TestUnifiedInjectDrift(unittest.TestCase):
    """Tests for the unified inject_drift() method."""

    def setUp(self):
        """Create sample data with mixed column types."""
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame(
            {
                # Numeric columns
                "age": np.random.randint(18, 80, n),
                "income": np.random.normal(50000, 15000, n),
                # Categorical columns
                "gender": np.random.choice(["M", "F", "Other"], n),
                "city": np.random.choice(["Madrid", "Barcelona", "Valencia"], n),
                # Boolean columns
                "is_active": np.random.choice([True, False], n),
                "flag": np.random.choice([0, 1], n),
            }
        )
        self.injector = DriftInjector(auto_report=False, random_state=42)

    def test_unified_drift_abrupt(self):
        """Test unified drift with abrupt mode."""
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["age", "income"],
            drift_mode="abrupt",
            drift_magnitude=0.3,
            start_index=100,
        )

        # Check that drift was applied to second half
        first_half_age = drifted_df.iloc[:100]["age"]
        second_half_age = drifted_df.iloc[100:]["age"]

        # First half should be unchanged
        pd.testing.assert_series_equal(
            first_half_age, self.df.iloc[:100]["age"], check_names=False
        )

        # Second half should be different (on average)
        self.assertNotEqual(second_half_age.mean(), self.df.iloc[100:]["age"].mean())

    def test_unified_drift_gradual(self):
        """Test unified drift with gradual mode."""
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["income"],
            drift_mode="gradual",
            drift_magnitude=0.5,
            center=100,
            width=50,
            profile="sigmoid",
        )

        # Values should change gradually around the center
        self.assertEqual(len(drifted_df), len(self.df))
        # Income values should be modified
        self.assertFalse(self.df["income"].equals(drifted_df["income"]))

    def test_unified_drift_incremental(self):
        """Test unified drift with incremental mode."""
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["age"],
            drift_mode="incremental",
            drift_magnitude=0.2,
        )

        self.assertEqual(len(drifted_df), len(self.df))
        self.assertFalse(self.df["age"].equals(drifted_df["age"]))

    def test_unified_drift_recurrent(self):
        """Test unified drift with recurrent mode."""
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["income"],
            drift_mode="recurrent",
            drift_magnitude=0.4,
            repeats=3,
        )

        self.assertEqual(len(drifted_df), len(self.df))
        self.assertFalse(self.df["income"].equals(drifted_df["income"]))

    def test_unified_drift_mixed_column_types(self):
        """Test that unified drift handles all column types correctly."""
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["age", "gender", "is_active"],  # Numeric, categorical, boolean
            drift_mode="abrupt",
            drift_magnitude=0.3,
            start_index=100,
        )

        self.assertEqual(len(drifted_df), len(self.df))
        # All three columns should potentially be affected
        self.assertFalse(self.df["age"].iloc[100:].equals(drifted_df["age"].iloc[100:]))

    def test_unified_drift_auto_detects_types(self):
        """Test that column types are correctly auto-detected."""
        # This should work without specifying operations
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["age", "income", "gender", "city", "is_active", "flag"],
            drift_mode="abrupt",
            drift_magnitude=0.2,
            start_index=150,
        )

        self.assertEqual(len(drifted_df), len(self.df))

    def test_unified_drift_custom_operations(self):
        """Test that custom operations per column type work."""
        drifted_df = self.injector.inject_drift(
            df=self.df,
            columns=["age", "gender"],
            drift_mode="abrupt",
            drift_magnitude=0.3,
            numeric_operation="scale",
            categorical_operation="frequency",
            start_index=100,
        )

        self.assertEqual(len(drifted_df), len(self.df))

    def test_unified_drift_invalid_mode_raises_error(self):
        """Test that invalid drift_mode raises ValueError."""
        with self.assertRaises(ValueError):
            self.injector.inject_drift(
                df=self.df,
                columns=["age"],
                drift_mode="invalid_mode",
                drift_magnitude=0.3,
            )

    def test_unified_drift_missing_column_warning(self):
        """Test that missing columns generate warnings but don't fail."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            drifted_df = self.injector.inject_drift(
                df=self.df,
                columns=["age", "nonexistent_column"],
                drift_mode="abrupt",
                drift_magnitude=0.3,
            )

            # Should have generated a warning
            self.assertTrue(
                any("nonexistent_column" in str(warning.message) for warning in w)
            )
            # But should still return valid data
            self.assertEqual(len(drifted_df), len(self.df))


if __name__ == "__main__":
    unittest.main()
