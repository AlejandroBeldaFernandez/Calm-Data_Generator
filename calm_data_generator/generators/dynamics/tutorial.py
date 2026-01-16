import pandas as pd
import numpy as np
from calm_data_generator.generators.dynamics.DynamicsInjector import DynamicsInjector
import os


def run_tutorial():
    print("=== DynamicsInjector Tutorial ===")

    # 1. Create Base Data
    print("\nGenerating base dataset...")
    n_samples = 1000
    df = pd.DataFrame(
        {
            "time": np.arange(n_samples),
            "feature_A": np.random.normal(10, 2, n_samples),
            "feature_B": np.random.uniform(0, 1, n_samples),
        }
    )
    print("Original Data Head:\n", df.head())

    # 2. Initialize Injector
    injector = DynamicsInjector(seed=42)

    # 3. Evolve Features
    print("\nEvolving features...")
    # Feature A: Linear trend (slope 0.01)
    # Feature B: Sinusoidal cycle (period 200, amplitude 0.5)
    evolution_config = {
        "feature_A": {"type": "linear", "slope": 0.01},
        "feature_B": {"type": "cycle", "period": 200, "amplitude": 0.5},
    }

    df_evolved = injector.evolve_features(df, evolution_config, time_col="time")

    # 4. Construct Target
    print("\nConstructing target variable...")
    # Target depends on evolved features: 0.5*A + 2*B
    # Classification task with threshold 8.0
    df_final = injector.construct_target(
        df_evolved,
        target_col="target",
        formula="0.5 * feature_A + 2.0 * feature_B",
        task_type="classification",
        threshold=8.0,
        noise_std=0.1,
    )

    # 5. Save and Verify
    output_dir = "dynamics_tutorial_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evolved_data.csv")
    df_final.to_csv(output_path, index=False)

    print(f"\nData saved to {output_path}")
    print("Final Data Head:\n", df_final.head())
    print("Target Balance:\n", df_final["target"].value_counts(normalize=True))


if __name__ == "__main__":
    run_tutorial()
