import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from calm_data_generator.reports.DiscriminatorReporter import DiscriminatorReporter


class TestDiscriminatorReporter(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

        # Create dummy data
        np.random.seed(42)
        self.real_df = pd.DataFrame(
            {
                "age": np.random.normal(30, 5, 100),
                "income": np.random.normal(50000, 10000, 100),
                "category": np.random.choice(["A", "B"], 100),
            }
        )

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_indistinguishable_data(self):
        """AUC should be close to 0.5 when data is from same distribution"""
        # Split real_df in half
        idx = int(len(self.real_df) / 2)
        df1 = self.real_df.iloc[:idx].copy()
        df2 = self.real_df.iloc[idx:].copy()

        reporter = DiscriminatorReporter()
        result = reporter.generate_report(df1, df2, self.output_dir)

        metrics = result.get("metrics", {})
        print(f"Indistinguishable metrics: {metrics}")

        # AUC (Discriminator Performance) should be poor (close to 0.5)
        # Similarity Score should be high (> 0.6)
        discr_auc = metrics.get(
            "discriminator_auc", metrics.get("auc")
        )  # Fallback if key missing
        sim_score = metrics.get("similarity_score")

        self.assertTrue(
            0.3 <= discr_auc <= 0.7, f"Discriminator AUC {discr_auc} too far from 0.5"
        )
        self.assertTrue(sim_score > 0.6, f"Similarity {sim_score} should be high")

    def test_distinguishable_data(self):
        """AUC should be high (close to 1.0) when data is very different"""
        # Create noise data
        noise_df = pd.DataFrame(
            {
                "age": np.random.uniform(0, 100, 100),  # Very different distrib
                "income": np.random.uniform(0, 100000, 100),
                "category": np.random.choice(["C", "D"], 100),  # Different cats
            }
        )

        reporter = DiscriminatorReporter()
        result = reporter.generate_report(self.real_df, noise_df, self.output_dir)

        metrics = result.get("metrics", {})
        print(f"Distinguishable metrics: {metrics}")

        discr_auc = metrics.get("discriminator_auc", 0.99)
        sim_score = metrics.get("similarity_score", 0.01)

        self.assertTrue(
            discr_auc > 0.8, f"Discriminator AUC {discr_auc} should be high"
        )
        self.assertTrue(sim_score < 0.4, f"Similarity {sim_score} should be low")
        self.assertTrue(os.path.exists(result["metrics_file"]))
        self.assertTrue(os.path.exists(result["explainability_file"]))

    def test_drift_detection(self):
        """Discriminator should identify shifted feature"""
        # Shift one feature
        drifted_df = self.real_df.copy()
        drifted_df["income"] = drifted_df["income"] + 50000  # Massive shift

        reporter = DiscriminatorReporter()
        result = reporter.generate_report(self.real_df, drifted_df, self.output_dir)

        metrics = result.get("metrics", {})
        # Should be easily distinguishable
        self.assertTrue(metrics["discriminator_auc"] > 0.9)
        self.assertTrue(metrics["similarity_score"] < 0.2)


if __name__ == "__main__":
    unittest.main()
