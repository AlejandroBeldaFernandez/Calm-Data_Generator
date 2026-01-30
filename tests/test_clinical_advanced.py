import os
import tempfile
import unittest
import numpy as np
import shutil
from calm_data_generator.generators.clinical.Clinic import ClinicalDataGenerator


class TestClinicalAdvanced(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        np.random.seed(42)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_clinical_generation_flow(self):
        """Test full Clinical Generator flow."""
        try:
            generator = ClinicalDataGenerator(seed=42)

            # ClinicalDataGenerator generates a dictionary of components
            # using its own synthetic logic (not necessarily from a template df)
            results = generator.generate(
                n_samples=50, n_genes=100, n_proteins=50, save_dataset=False
            )

            self.assertIsNotNone(results)
            self.assertIn("demographics", results)
            self.assertIn("genes", results)
            self.assertIn("proteins", results)

            demo_df = results["demographics"]
            self.assertEqual(len(demo_df), 50)
            self.assertIn("Group", demo_df.columns)

        except ImportError as e:
            print(
                f"Skipping test_clinical_generation_flow due to missing dependency: {e}"
            )
        except Exception as e:
            self.fail(f"Clinical generator failed: {e}")


if __name__ == "__main__":
    unittest.main()
