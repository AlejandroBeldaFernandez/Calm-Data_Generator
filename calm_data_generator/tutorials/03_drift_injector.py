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
# 2. Gradual Feature Drift
# ============================================================

# Initialize DriftInjector
injector = DriftInjector(time_col="timestamp")

# Inject gradual drift using transition window
# Available drift_types: gaussian_noise, shift, scale, add_value, subtract_value
drifted_gradual = injector.inject_feature_drift_gradual(
    df=data.copy(),
    feature_cols=["feature1"],
    drift_magnitude=0.5,  # Magnitude of the drift
    drift_type="shift",  # Type: gaussian_noise, shift, scale
    start_index=50,  # Start drift at row 50
    center=25,  # Center of transition window (relative to selected rows)
    width=20,  # Width of transition window
    profile="sigmoid",  # Transition profile: sigmoid, linear, cosine
    auto_report=False,  # Disable automatic reporting
)

print("\n--- Gradual Feature Drift ---")
print(f"Original feature1 mean (last 20): {data['feature1'].tail(20).mean():.2f}")
print(
    f"Drifted feature1 mean (last 20): {drifted_gradual['feature1'].tail(20).mean():.2f}"
)

# ============================================================
# 3. Abrupt Feature Drift (Single Point)
# ============================================================

# Use inject_feature_drift for immediate/abrupt changes
drifted_abrupt = injector.inject_feature_drift(
    df=data.copy(),
    feature_cols=["feature2"],
    drift_magnitude=0.8,
    drift_type="shift",
    start_index=60,  # Apply drift from row 60 onwards
    auto_report=False,
)

print("\n--- Abrupt Feature Drift ---")
print(f"Feature2 mean before (0-59): {drifted_abrupt['feature2'].iloc[:60].mean():.2f}")
print(f"Feature2 mean after (60-99): {drifted_abrupt['feature2'].iloc[60:].mean():.2f}")

# ============================================================
# 4. Scale Drift (Change Variance)
# ============================================================

drifted_scale = injector.inject_feature_drift_gradual(
    df=data.copy(),
    feature_cols=["feature1"],
    drift_magnitude=1.5,  # Scale factor
    drift_type="scale",  # Scales the data
    start_index=0,
    center=50,
    width=80,
    auto_report=False,
)

print("\n--- Scale Drift ---")
print(f"Original std: {data['feature1'].std():.2f}")
print(f"Drifted std: {drifted_scale['feature1'].std():.2f}")

# ============================================================
# 5. Label Drift
# ============================================================

drifted_labels = injector.inject_label_drift_gradual(
    df=data.copy(),
    target_col="target",
    drift_magnitude=0.3,  # 30% probability of label flip
    start_index=70,
    center=15,
    width=10,
    auto_report=False,
)

print("\n--- Label Drift ---")
print(f"Original target mean (last 30): {data['target'].tail(30).mean():.2f}")
print(f"Drifted target mean (last 30): {drifted_labels['target'].tail(30).mean():.2f}")

# ============================================================
# 6. Conditional Drift
# ============================================================

# Apply drift only to rows meeting certain conditions
conditions = [{"column": "feature1", "operator": ">", "value": 50}]

drifted_conditional = injector.inject_conditional_drift(
    df=data.copy(),
    feature_cols=["feature2"],
    conditions=conditions,
    drift_type="shift",
    drift_magnitude=0.5,
    auto_report=False,
)

print("\n--- Conditional Drift ---")
print("Drift applied only where feature1 > 50")

# ============================================================
# 7. Outlier Injection
# ============================================================

drifted_outliers = injector.inject_outliers_global(
    df=data.copy(),
    cols=["feature1", "feature2"],
    outlier_prob=0.05,  # 5% of rows become outliers
    factor=3.0,  # Outliers are 3x the standard value
    auto_report=False,
)

print("\n--- Outlier Injection ---")
print(f"Original max feature1: {data['feature1'].max():.2f}")
print(f"After outliers max feature1: {drifted_outliers['feature1'].max():.2f}")

# ============================================================
# 8. Inject Multiple Issues at Once
# ============================================================

issues = [
    {
        "issue_type": "feature_drift",
        "cols": ["feature1"],
        "params": {"drift_type": "shift", "drift_magnitude": 0.3, "start_index": 50},
    },
    {
        "issue_type": "outliers",
        "cols": ["feature2"],
        "params": {"outlier_prob": 0.02, "factor": 4.0},
    },
]

drifted_combined = injector.inject_data_quality_issues(
    df=data.copy(), issues=issues, auto_report=False
)

print("\n--- Combined Data Quality Issues ---")
print("Multiple drift types applied in sequence!")

print("\n✅ DriftInjector tutorial completed!")
print("\nAvailable methods:")
print("  - inject_feature_drift: Immediate/abrupt feature drift")
print("  - inject_feature_drift_gradual: Smooth transition drift")
print("  - inject_feature_drift_incremental: Constant smooth drift")
print("  - inject_feature_drift_recurrent: Recurring drift windows")
print("  - inject_label_drift: Random label flips")
print("  - inject_label_drift_gradual: Gradual label changes")
print("  - inject_conditional_drift: Drift based on conditions")
print("  - inject_outliers_global: Global outlier injection")
print("  - inject_new_value: Inject new categorical values")
print("  - inject_data_quality_issues: Multiple issues at once")
