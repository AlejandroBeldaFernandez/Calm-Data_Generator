import unittest
import pandas as pd
import shutil
import os
import pytest

try:
    from river import synth

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    synth = None

from calm_data_generator.generators.clinical.ClinicGeneratorBlock import (
    ClinicalDataGeneratorBlock,
)


@pytest.mark.skipif(not RIVER_AVAILABLE, reason="River/Synth not installed")
class TestClinicalBlockGenerator(unittest.TestCase):
    def setUp(self):
        self.output_dir = "test_output_clinic_block"
        os.makedirs(self.output_dir, exist_ok=True)
        self.filename = "test_clinic_blocks.csv"

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_initialization(self):
        gen = ClinicalDataGeneratorBlock()
        self.assertIsInstance(gen, ClinicalDataGeneratorBlock)

    def test_generate_clinical_blocks(self):
        if not RIVER_AVAILABLE:
            pytest.skip("River not available")

        gen = ClinicalDataGeneratorBlock()
        river_gen = synth.Agrawal(seed=42)

        full_path = gen.generate(
            output_dir=self.output_dir,
            filename=self.filename,
            n_blocks=2,
            total_samples=20,
            n_samples_block=[10, 10],
            generators=[river_gen, river_gen],
            target_col="diagnosis",
            generate_report=False,
        )

        self.assertTrue(os.path.exists(full_path))
        df = pd.read_csv(full_path)
        self.assertEqual(len(df), 20)
        self.assertTrue("block" in df.columns)
        self.assertEqual(set(df["block"].unique()), {1, 2})
        self.assertTrue("Age" in df.columns)


if __name__ == "__main__":
    unittest.main()
