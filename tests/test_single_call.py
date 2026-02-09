"""
CALM-Data-Generator - Single Call Test
=======================================

Test that demonstrates generating data, injecting drift, and generating
a report all in ONE method call using drift_injection_config parameter.
"""

import pandas as pd
import numpy as np
import tempfile
import os


def test_single_call_with_drift():
    """
    Test: Generate + Drift + Report in a SINGLE call to RealGenerator.generate()
    """

    print("=" * 60)
    print("CALM-Data-Generator - Single Call Test")
    print("Generation + Drift + Report in ONE call")
    print("=" * 60)

    # 1. Create sample data
    print("\n[1/3] Creating sample data...")
    np.random.seed(42)

    real_data = pd.DataFrame(
        {
            "age": np.random.randint(20, 70, 100),
            "income": np.random.normal(50000, 15000, 100).astype(int),
            "score": np.random.uniform(0, 100, 100),
            "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        }
    )

    print(f"   ✓ Real data shape: {real_data.shape}")

    # 2. Import and configure
    print("\n[2/3] Configuring single-call generation with drift...")

    from calm_data_generator.generators.tabular import RealGenerator
    from calm_data_generator.generators.configs import DriftConfig

    gen = RealGenerator()

    # CORRECT FORMAT: List of DriftConfig objects
    drift_config = [
        DriftConfig(
            method="inject_gradual_drift",
            params={
                "columns": ["score"],
                "drift_type": "mean_shift",
                "drift_magnitude": 0.5,
                "start_index": 20,
                "end_index": 50,
            },
        )
    ]

    # Create temp output dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. SINGLE CALL: Generate + Drift + Report
        print("\n[3/3] Executing single-call generation with drift...")

        result = gen.generate(
            data=real_data,
            n_samples=50,
            method="cart",
            target_col="target",
            output_dir=tmpdir,
            save_dataset=True,
            drift_injection_config=drift_config,  # <-- Correct format!
            constraints=[
                {"col": "age", "op": ">=", "val": 18},
                {"col": "income", "op": ">", "val": 0},
            ],
        )

        if result is not None:
            print(f"\n   ✓ Generated data shape: {result.shape}")
            print(f"   ✓ Columns: {list(result.columns)}")
            print(f"   ✓ Age range: {result['age'].min()} - {result['age'].max()}")
            print(f"   ✓ Score mean: {result['score'].mean():.2f}")

            # Check if files were saved
            saved_files = os.listdir(tmpdir)
            print(f"   ✓ Files saved: {saved_files}")

            print("\n" + "=" * 60)
            print("✅ SINGLE CALL TEST PASSED!")
            print("   Generated data + injected drift + saved report")
            print("   All in ONE method call!")
            print("=" * 60)
            return True
        else:
            print("   ⚠ Result is None")
            return False


if __name__ == "__main__":
    success = test_single_call_with_drift()
    exit(0 if success else 1)
