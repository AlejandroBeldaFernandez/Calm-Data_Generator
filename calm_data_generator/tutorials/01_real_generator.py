"""
Tutorial 1: RealGenerator - Synthetic Data from Real Datasets
==============================================================

This tutorial demonstrates how to use RealGenerator to create
synthetic data that preserves the statistical properties of real data.
"""

import pandas as pd
import numpy as np
from calm_data_generator.generators.real import RealGenerator

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

# Generate synthetic data using CART (fast, tree-based)
synthetic = gen.generate(data=data, n_samples=200, method="cart", target_col="target")

print("\nSynthetic data shape:", synthetic.shape)
print(synthetic.head())

# ============================================================
# 2. Deep Learning Methods - CTGAN/TVAE
# ============================================================

# Generate with CTGAN (requires torch)
try:
    synthetic_ctgan = gen.generate(
        data=data, n_samples=100, method="ctgan", model_params={"sdv_epochs": 100}
    )
    print("\nCTGAN synthetic data:", synthetic_ctgan.shape)
except Exception as e:
    print(f"CTGAN not available: {e}")

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
# 4. Oversampling Methods - SMOTE/ADASYN
# ============================================================

# Balance classes with SMOTE
synthetic_smote = gen.generate(
    data=data, n_samples=100, method="smote", target_col="target"
)

print("\nSMOTE class distribution:")
print(synthetic_smote["target"].value_counts())

# ============================================================
# 5. Time Series - Temporal Copula
# ============================================================

# Create time series data
ts_data = pd.DataFrame(
    {
        "entity_id": [f"E_{i // 5}" for i in range(50)],
        "timestep": list(range(5)) * 10,
        "value": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    }
)

synthetic_ts = gen.generate(
    data=ts_data,
    n_samples=30,
    method="copula_temporal",
    block_column="entity_id",
    model_params={"time_col": "timestep"},
)

print("\nTemporal Copula result:", synthetic_ts.shape)

print("\n✅ Tutorial completed!")
