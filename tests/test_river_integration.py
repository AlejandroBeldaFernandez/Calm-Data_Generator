import unittest
import pandas as pd
import numpy as np
import shutil
import tempfile
import os
import pytest

# Try to import river to skip if not available
try:
    from river import synth

    RIVER_AVAILABLE = True
except ImportError:
    try:
        from river.datasets import synth

        RIVER_AVAILABLE = True
    except ImportError:
        RIVER_AVAILABLE = False
        synth = None

from calm_data_generator.generators.stream.StreamGenerator import StreamGenerator


class TestRiverIntegration(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        np.random.seed(42)

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    def test_river_drift_detection(self):
        """Test StreamGenerator with River synth generator."""

        if not RIVER_AVAILABLE:
            pytest.skip("River not available")

        # Create a River generator
        agrawal_gen = synth.Agrawal(seed=42)

        generator = StreamGenerator(auto_report=False)

        try:
            # Generate from River stream
            n_samples = 100
            synthetic_stream = generator.generate(
                generator_instance=agrawal_gen,
                n_samples=n_samples,
                output_dir=self.output_dir,
            )

            self.assertIsNotNone(synthetic_stream)
            self.assertEqual(len(synthetic_stream), n_samples)

            # Check if columns are present (Agrawal features: salary, commission, age, etc.)
            self.assertIn("target", synthetic_stream.columns)
            self.assertTrue(len(synthetic_stream.columns) > 1)

            print("River integration test passed (River Generator)")

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.fail(f"StreamGenerator failed with River available: {e}")


if __name__ == "__main__":
    unittest.main()
