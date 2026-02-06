import pandas as pd
import numpy as np
import logging
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator

# Configure logging to show everything
logging.basicConfig(level=logging.DEBUG)


def debug_small_sample_synthesis():
    # Create small dataset like in the failing test (50 samples)
    np.random.seed(42)
    n_samples = 50
    data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n_samples),
            "feature2": np.random.choice(["A", "B"], n_samples),
            # Target column
            "target": np.random.randint(0, 2, n_samples),
        }
    )

    print("Created small dataset shape:", data.shape)

    gen = RealGenerator(auto_report=False)

    try:
        print("Starting generation...")
        synth = gen.generate(
            data=data,
            method="cart",
            target_col="target",
            n_samples=n_samples,
            output_dir="debug_output",
        )

        if synth is None:
            print("FAILURE: RealGenerator returned None")
        else:
            print(f"SUCCESS: Generated {len(synth)} samples")
            print(synth.head())

    except Exception as e:
        print(f"EXCEPTION CAUGHT IN SCRIPT: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_small_sample_synthesis()
