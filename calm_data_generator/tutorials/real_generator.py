"""
Tutorial 1: RealGenerator - Synthetic Data from Real Datasets
==============================================================

This tutorial demonstrates how to use RealGenerator to create
synthetic data that preserves the statistical properties of real data.
"""

import pandas as pd
import numpy as np
from calm_data_generator import RealGenerator
from calm_data_generator.generators.configs import DriftConfig, ReportConfig

# ============================================================
# 1. Basic Usage - Generate synthetic data with CART
# ============================================================

# Create sample dataset
# NOTE: In a real-world scenario, you would load your own dataset here.
# Example: data = pd.read_csv("path/to/your_real_data.csv")
np.random.seed(42)
data = pd.DataFrame(
    {
        "age": np.random.randint(18, 80, 100),
        "income": np.random.normal(50000, 15000, 100),
        "score": np.random.uniform(0, 1, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
    }
)

print("Original data shape:", data.shape)
print(data.head())

# Initialize generator
gen = RealGenerator()

# Define Configurations (New API)
drift_config = [
    DriftConfig(
        method="inject_feature_drift",
        params={"feature_cols": ["income"], "drift_magnitude": 0.5},
    )
]
report_config = ReportConfig(output_dir="tutorial_output", target_column="target")

# Generate synthetic data using CART (fast, tree-based)
# Passing configs to enable automatic drift injection and reporting
synthetic = gen.generate(
    data=data,
    n_samples=200,
    method="cart",
    target_col="target",
    drift_injection_config=drift_config,
    report_config=report_config,
    auto_report=True,
)

print("\nSynthetic data shape:", synthetic.shape)
print(synthetic.head())

# ============================================================
# 2. Deep Learning Methods - CTGAN/TVAE
# ============================================================

# Generate with CTGAN (requires synthcity/torch)
try:
    synthetic_ctgan = gen.generate(
        data=data,
        n_samples=100,
        method="ctgan",
        epochs=100,  # Passed as kwargs
        batch_size=64,
    )
    print("\nCTGAN synthetic data:", synthetic_ctgan.shape)
except Exception as e:
    print(f"CTGAN not available: {e}")

# Generate with Gaussian Copula
try:
    synthetic_copula = gen.generate(data=data, n_samples=100, method="copula")
    print("\nCopula synthetic data:", synthetic_copula.shape)
except Exception as e:
    print(f"Copula failed: {e}")


# ============================================================
# 3. Constraints - Apply business rules
# ============================================================

# Generate with constraints: age > 20, income > 0
constraints = [
    {"col": "age", "op": ">", "val": 20},
    {"col": "income", "op": ">", "val": 0},
]

synthetic_constrained = gen.generate(
    data=data, n_samples=100, method="cart", constraints=constraints
)

print("\nConstrained data - Min age:", synthetic_constrained["age"].min())
print("Constrained data - Min income:", synthetic_constrained["income"].min())

# ============================================================
# 4. Single-Cell - scVI & GEARS
# ============================================================

# Generate with scVI
try:
    synthetic_scvi = gen.generate(data=data, n_samples=50, method="scvi", epochs=10)
    print("\nscVI synthetic data:", synthetic_scvi.shape)
except Exception as e:
    print(f"scVI not available: {e}")

# Generate with GEARS (Perturbation Prediction)
try:
    synthetic_gears = gen.generate(
        data=data,
        n_samples=50,
        method="gears",
        perturbations=["age", "income"],  # Genes/features to perturb
        epochs=10,
    )
    print("\nGEARS synthetic data:", synthetic_gears.shape)
except Exception as e:
    print(f"GEARS not available: {e}")


# ============================================================
# 5. Oversampling Methods - SMOTE/ADASYN
# ============================================================


# Balance classes with SMOTE
synthetic_smote = gen.generate(
    data=data, n_samples=100, method="smote", target_col="target"
)

print("\nSMOTE class distribution:")
print(synthetic_smote["target"].value_counts())

# ============================================================
# 6. Advanced Configuration (Drift & Reporting) - REMOVED AS REDUNDANT
# ============================================================
# The advanced configuration for Drift and Reporting has been integrated
# into the "1. Basic Usage" section to streamline the tutorial.
# The previous section 6 content is now redundant.
#
# print(f"Advanced synthetic data shape: {synthetic_advanced.shape}")
# print(f"Check 'tutorial_real_output' for the generated report.")
