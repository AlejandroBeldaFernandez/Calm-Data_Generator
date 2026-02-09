import unittest
import pandas as pd
import numpy as np
import shutil
import os
from calm_data_generator.generators.tabular.RealBlockGenerator import RealBlockGenerator
from calm_data_generator.generators.configs import DriftConfig, ReportConfig


class TestRealBlockGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_real_block"
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a sample dataset
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n_samples),
                "feature2": np.random.choice(["A", "B"], n_samples),
                "block_col": np.repeat(["Block1", "Block2"], n_samples // 2),
                "target": np.random.randint(0, 2, n_samples),
            }
        )

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_initialization(self):
        gen = RealBlockGenerator()
        self.assertIsInstance(gen, RealBlockGenerator)

    def test_generate_with_existing_block_column(self):
        gen = RealBlockGenerator(auto_report=False)
        synthetic_data = gen.generate(
            data=self.data,
            output_dir=self.output_dir,
            method="cart",
            block_column="block_col",
            target_col="target",
        )

        self.assertIsInstance(synthetic_data, pd.DataFrame)
        self.assertTrue("block_col" in synthetic_data.columns)
        self.assertEqual(
            set(synthetic_data["block_col"].unique()), {"Block1", "Block2"}
        )
        self.assertFalse(synthetic_data.empty)

    def test_generate_with_chunk_size(self):
        gen = RealBlockGenerator(auto_report=False)
        # Drop block column to test chunking
        data_no_block = self.data.drop(columns=["block_col"])

        chunk_size = 20
        synthetic_data = gen.generate(
            data=data_no_block,
            output_dir=self.output_dir,
            method="cart",
            chunk_size=chunk_size,
            target_col="target",
        )

        self.assertTrue("chunk" in synthetic_data.columns)
        # 100 samples / 20 chunk size = 5 chunks (0 to 4)
        self.assertEqual(len(synthetic_data["chunk"].unique()), 5)

    def test_generate_with_n_samples_block_dict(self):
        gen = RealBlockGenerator(auto_report=False)

        n_samples_map = {"Block1": 10, "Block2": 20}

        synthetic_data = gen.generate(
            data=self.data,
            output_dir=self.output_dir,
            method="cart",
            block_column="block_col",
            n_samples_block=n_samples_map,
            target_col="target",
        )

        counts = synthetic_data["block_col"].value_counts()
        self.assertEqual(counts["Block1"], 10)
        self.assertEqual(counts["Block2"], 20)

    def test_generate_with_config_objects(self):
        """Test generation with DriftConfig and ReportConfig objects."""
        gen = RealBlockGenerator(auto_report=False)

        drift_conf = DriftConfig(
            method="inject_feature_drift",
            params={"feature_cols": ["feature1"], "drift_magnitude": 0.5},
        )
        report_conf = ReportConfig(output_dir=self.output_dir, target_column="target")

        synthetic_data = gen.generate(
            data=self.data,
            output_dir=self.output_dir,
            method="cart",
            block_column="block_col",
            drift_config=[drift_conf],
            report_config=report_conf,
            target_col="target",
        )

        self.assertFalse(synthetic_data.empty)
        # Verify ReportConfig usage implicitly by checking if report generation didn't crash
        # (RealBlockGenerator uses report_config to pass to reporter)


if __name__ == "__main__":
    unittest.main()
