"""
Tutorial 3: DriftInjector - Injecting Drift for ML Testing
============================================================

This tutorial demonstrates how to inject various types of drift
into datasets for testing machine learning model robustness.

DriftInjector supports:
- Feature drift (gradual, incremental, recurrent)
- Label drift
- Outlier injection
- Conditional drift based on rules
"""

import pandas as pd
import numpy as np
from calm_data_generator.generators.drift import DriftInjector
from calm_data_generator.generators.configs import DriftConfig

# ============================================================
# 1. Create sample data
# ============================================================

np.random.seed(42)
data = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
        "feature1": np.random.normal(50, 10, 100),
        "feature2": np.random.normal(100, 20, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100),
    }
)

print("Original data:")
print(data.head())
print(f"\nFeature1 mean: {data['feature1'].mean():.2f}")
print(f"Feature2 mean: {data['feature2'].mean():.2f}")

# ============================================================
# 2. Define Drift Configurations (New API)
# ============================================================

# The new API uses DriftConfig objects to define drift parameters.
# This allows for serializable, validatable, and reusable configurations.

# Gradual Feature Drift
config_gradual = DriftConfig(
    method="inject_feature_drift_gradual",
    params={
        "feature_cols": ["feature1"],
        "drift_magnitude": 0.5,
        "drift_type": "shift",
        "start_index": 50,
        "center": 25,
        "width": 20,
        "profile": "sigmoid",
    },
)

# Abrupt Feature Drift
config_abrupt = DriftConfig(
    method="inject_feature_drift",
    params={
        "feature_cols": ["feature2"],
        "drift_magnitude": 0.8,
        "drift_type": "shift",
        "start_index": 60,
    },
)

# Label Drift
config_label = DriftConfig(
    method="inject_label_drift_gradual",
    params={
        "target_col": "target",
        "drift_magnitude": 0.3,
        "start_index": 70,
        "center": 15,
        "width": 10,
    },
)

# Outliers
config_outliers = DriftConfig(
    method="inject_outliers_global",
    params={"cols": ["feature1", "feature2"], "outlier_prob": 0.05, "factor": 3.0},
)

# ============================================================
# 3. Apply Drift using DriftInjector
# ============================================================

print("\n--- Applying Drifts via Configuration ---")

# Initialize Injector
injector = DriftInjector(output_dir="drift_tutorial_output", time_col="timestamp")

# Create a schedule of drifts
drift_schedule = [
    config_gradual,
    config_abrupt,
    config_label,
    # config_outliers # Uncomment to include outliers
]

# Apply all drifts in the schedule
drifted_data = injector.inject_multiple_types_of_drift(
    df=data.copy(),
    schedule=drift_schedule,
    time_col="timestamp",
    target_column="target",
)

# ============================================================
# 4. Verify Results
# ============================================================

print("\n--- Results Analysis ---")

print(f"Original feature1 mean (last 20): {data['feature1'].tail(20).mean():.2f}")
print(
    f"Drifted feature1 mean (last 20): {drifted_data['feature1'].tail(20).mean():.2f}"
)

print(f"\nOriginal feature2 mean (last 40): {data['feature2'].tail(40).mean():.2f}")
print(
    f"Drifted feature2 mean (last 40): {drifted_data['feature2'].tail(40).mean():.2f}"
)

print("\n✅ DriftInjector tutorial completed (Config API)!")
