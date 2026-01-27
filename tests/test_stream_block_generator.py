import unittest
import pandas as pd
import shutil
import os
from river import synth
from calm_data_generator.generators.stream.StreamBlockGenerator import (
    SyntheticBlockGenerator,
)


class TestSyntheticBlockGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_stream_block"
        os.makedirs(self.output_dir, exist_ok=True)
        self.filename = "test_stream_blocks.csv"

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_initialization(self):
        gen = SyntheticBlockGenerator()
        self.assertIsInstance(gen, SyntheticBlockGenerator)

    def test_generate_simple_interface(self):
        gen = SyntheticBlockGenerator()
        path = gen.generate_blocks_simple(
            output_dir=self.output_dir,
            filename=self.filename,
            n_blocks=2,
            total_samples=100,
            methods=["sea"],  # Use SEA generator
            method_params=[{"seed": 42}, {"seed": 43}],
            generate_report=False,
        )

        self.assertTrue(os.path.exists(path))
        df = pd.read_csv(path)
        self.assertEqual(len(df), 100)
        self.assertTrue("block" in df.columns)
        self.assertEqual(set(df["block"].unique()), {1, 2})

    def test_generate_manual_interface(self):
        gen = SyntheticBlockGenerator()

        # Create different River generators to simulate concept drift
        gen1 = synth.Agrawal(seed=42, classification_function=0)
        gen2 = synth.Agrawal(seed=42, classification_function=1)  # Different concept

        path = gen.generate(
            output_dir=self.output_dir,
            filename=self.filename,
            n_blocks=2,
            total_samples=50,
            n_samples_block=[25, 25],
            generators=[gen1, gen2],
            generate_report=False,
        )

        df = pd.read_csv(path)
        self.assertEqual(len(df), 50)
        self.assertEqual(df["block"].value_counts()[1], 25)
        self.assertEqual(df["block"].value_counts()[2], 25)


if __name__ == "__main__":
    unittest.main()
