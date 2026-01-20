import pandas as pd
import numpy as np
from calm_data_generator.generators.drift.DriftInjector import DriftInjector
import os


def run_tutorial():
    print("=== DriftInjector Tutorial ===")

    # 1. Create Synthetic Data
    print("\nGenerating base dataset...")
    n_samples = 1000
    df = pd.DataFrame(
        {
            "feature_A": np.random.normal(10, 2, n_samples),
            "feature_B": np.random.uniform(0, 1, n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )
    print("Original Data Stats:\n", df.describe().loc[["mean", "std"]])

    # 2. Initialize Injector
    output_dir = "drift_tutorial_output"
    injector = DriftInjector(
        original_df=df, output_dir=output_dir, generator_name="tutorial_drift"
    )

    # 3. Inject Abrupt Feature Drift
    print("\nInjecting Abrupt Feature Drift (Shift) in feature_A...")
    # Shift mean by +5 starting from index 500
    df_drifted = injector.inject_feature_drift_abrupt(
        df=df,
        feature_cols=["feature_A"],
        drift_type="shift",
        drift_magnitude=5.0,
        change_index=500,
        direction="up",
    )

    # 4. Inject Gradual Feature Drift (Scale)
    print("\nInjecting Gradual Feature Drift (Scale) in feature_B...")
    # Scale variance by 2x gradually centered at 500 with width 600 (200 to 800)
    df_drifted = injector.inject_feature_drift_gradual(
        df=df_drifted,
        feature_cols=["feature_B"],
        drift_type="scale",
        drift_magnitude=2.0,
        center=500,
        width=600,
        profile="sigmoid",
    )

    # 5. Save and Verify
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "drifted_data.csv")
    df_drifted.to_csv(output_path, index=False)

    print(f"\nDrifted data saved to {output_path}")

    # Verification
    print("\nVerification:")
    print("Feature A Mean (0-500):", df_drifted["feature_A"].iloc[:500].mean())
    print("Feature A Mean (500-1000):", df_drifted["feature_A"].iloc[500:].mean())

    print("Feature B Std (0-200):", df_drifted["feature_B"].iloc[:200].std())
    print("Feature B Std (800-1000):", df_drifted["feature_B"].iloc[800:].std())


if __name__ == "__main__":
    run_tutorial()
