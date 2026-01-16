"""
Tutorial 3: DriftInjector - Injecting Drift for ML Testing
============================================================

This tutorial demonstrates how to inject various types of drift
into datasets for testing machine learning model robustness.
"""

import pandas as pd
import numpy as np
from calm_data_generator.generators.drift import DriftInjector

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
# 2. Gradual Drift (Covariate Shift)
# ============================================================

injector = DriftInjector(time_col="timestamp")

# Inject gradual drift to feature1 (mean shift over time)
drifted_gradual = injector.inject_gradual_drift(
    df=data.copy(),
    columns=["feature1"],
    drift_magnitude=0.5,  # 50% change
    drift_type="mean_shift",
    start_index=50,  # Start drift at row 50
    end_index=100,  # End at row 100
    transition_width=20,  # Gradual over 20 rows
)

print("\n--- Gradual Drift ---")
print(f"Original feature1 mean (last 20): {data['feature1'].tail(20).mean():.2f}")
print(
    f"Drifted feature1 mean (last 20): {drifted_gradual['feature1'].tail(20).mean():.2f}"
)

# ============================================================
# 3. Sudden Drift (Abrupt Change)
# ============================================================

drifted_sudden = injector.inject_sudden_drift(
    df=data.copy(),
    columns=["feature2"],
    drift_magnitude=1.0,  # 100% change
    drift_type="mean_shift",
    change_point=60,  # Abrupt change at row 60
)

print("\n--- Sudden Drift ---")
print(f"Feature2 mean before (0-59): {drifted_sudden['feature2'].iloc[:60].mean():.2f}")
print(f"Feature2 mean after (60-99): {drifted_sudden['feature2'].iloc[60:].mean():.2f}")

# ============================================================
# 4. Variance Drift
# ============================================================

drifted_variance = injector.inject_gradual_drift(
    df=data.copy(),
    columns=["feature1"],
    drift_magnitude=2.0,  # Double the variance
    drift_type="variance_shift",
    start_index=0,
    end_index=100,
)

print("\n--- Variance Drift ---")
print(f"Original variance: {data['feature1'].std():.2f}")
print(f"Drifted variance: {drifted_variance['feature1'].std():.2f}")

# ============================================================
# 5. Categorical Drift
# ============================================================

drifted_categorical = injector.inject_gradual_drift(
    df=data.copy(),
    columns=["category"],
    drift_magnitude=0.8,  # 80% probability of change
    drift_type="category_shift",
    start_index=70,
    end_index=100,
)

print("\n--- Categorical Drift ---")
print("Original distribution (last 30):")
print(data["category"].tail(30).value_counts())
print("\nDrifted distribution (last 30):")
print(drifted_categorical["category"].tail(30).value_counts())

# ============================================================
# 6. Combined Drift (Multiple Features)
# ============================================================

drifted_combined = injector.inject_drift_from_config(
    df=data.copy(),
    config=[
        {
            "columns": ["feature1"],
            "drift_type": "mean_shift",
            "magnitude": 0.3,
            "start": 40,
            "end": 80,
            "transition": "gradual",
        },
        {
            "columns": ["feature2"],
            "drift_type": "variance_shift",
            "magnitude": 1.5,
            "start": 60,
            "end": 100,
            "transition": "sudden",
        },
    ],
)

print("\n--- Combined Drift Applied ---")
print("Ready for ML robustness testing!")

print("\n✅ Drift injection tutorial completed!")
