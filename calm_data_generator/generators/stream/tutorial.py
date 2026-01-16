import os
from calm_data_generator.generators.stream.StreamGenerator import StreamGenerator
from river.datasets import synth
from calm_data_generator.generators.configs import DateConfig


def run_tutorial():
    print("=== StreamGenerator Tutorial (Unified API) ===")
    output_dir = "synthetic_tutorial_output"

    # factory methods for reproducibility
    def get_sea_1():
        return synth.SEA(seed=42, variant=0)

    def get_sea_2():
        return synth.SEA(seed=42, variant=1)

    gen = StreamGenerator(random_state=42)

    # 1. Simple Generation (No Drift)
    print("\n1. Generating Simple Stream...")
    df_simple = gen.generate(
        generator_instance=get_sea_1(),
        n_samples=500,
        output_dir=output_dir,
        save_dataset=True,  # Will save to output_dir/synthetic_data.csv (default name)
    )
    print("Simple shape:", df_simple.shape)

    # 2. Concept Drift (Abrupt)
    # Using Unified API kwargs for concept drift
    print("\n2. Generating Concept Drift (Abrupt SEA 0 -> SEA 1)...")

    df_concept = gen.generate(
        generator_instance=get_sea_1(),
        n_samples=1000,
        output_dir=output_dir,
        save_dataset=False,  # We'll save manually to avoid overwrite or specific name
        # Concept Drift Kwargs
        drift_type="abrupt",
        generator_instance_drift=get_sea_2(),
        drift_point=500,
    )

    # Save manually with specific name
    os.makedirs(output_dir, exist_ok=True)
    df_concept.to_csv(os.path.join(output_dir, "concept_drift.csv"), index=False)
    print("Concept Drift shape:", df_concept.shape)

    # 3. Feature Drift Injection (Using Config)
    # Injecting noise into column 'col_0' starting at index 500
    print("\n3. Injecting Feature Drift...")

    drift_config = [
        {
            "method": "inject_feature_drift",
            "params": {
                "feature_cols": ["col_0"],
                "drift_type": "gaussian_noise",
                "drift_magnitude": 2.0,
                "start_index": 500,
            },
        }
    ]

    df_feature = gen.generate(
        generator_instance=get_sea_1(),
        n_samples=1000,
        drift_injection_config=drift_config,
        date_config=DateConfig(start_date="2025-01-01"),
        output_dir=output_dir,
    )
    df_feature.to_csv(os.path.join(output_dir, "feature_drift.csv"), index=False)
    print("Feature Drift shape:", df_feature.shape)

    print(f"\nAll data saved to {output_dir}/")


if __name__ == "__main__":
    run_tutorial()
