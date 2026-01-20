import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular.RealGenerator import RealGenerator
import os


def run_tutorial():
    print("=== RealGenerator Tutorial ===")

    # 1. Create Dummy Real Data
    print("\nCreating dummy real dataset...")
    n_real = 500
    real_data = pd.DataFrame(
        {
            "age": np.random.normal(40, 10, n_real).astype(int),
            "income": np.random.lognormal(10, 1, n_real),
            "category": np.random.choice(["A", "B", "C"], n_real),
        }
    )
    print("Real Data Head:\n", real_data.head())

    # 2. Initialize and Fit Generator
    print("\nFitting RealGenerator (GaussianCopula)...")
    # Note: This requires sdv installed. If not, it might fail or use a fallback if implemented.
    # Assuming sdv is available or RealGenerator handles it.
    try:
        # Initialize Generator with original data and method
        # 'copula' corresponds to GaussianCopula in SDV
        generator = RealGenerator()

        # 3. Generate Synthetic Data
        print("\nGenerating 200 synthetic samples...")
        # The generate method handles fitting internally if needed
        output_dir = "real_tutorial_output"
        synthetic_data = generator.generate(
            data=real_data,
            method="copula",
            n_samples=200,
            output_dir=output_dir,
            save_dataset=True,
        )

        output_path = os.path.join(output_dir, "synthetic_data_copula.csv")

        print(f"\nSynthetic data saved to {output_path}")
        print("Synthetic Data Head:\n", synthetic_data.head())

        # Simple comparison
        print("\nComparison (Mean Age):")
        print("Real:", real_data["age"].mean())
        print("Synthetic:", synthetic_data["age"].mean())

    except ImportError:
        print("\nSDV library not found. RealGenerator requires 'sdv' to run.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    run_tutorial()
