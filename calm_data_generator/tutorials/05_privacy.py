"""
Tutorial 5: Anonymizer - Privacy-Preserving Transformations
================================================================

This tutorial demonstrates how to apply privacy-preserving
transformations to protect sensitive data.
"""

import pandas as pd
import numpy as np
from calm_data_generator.anonymizer import (
    pseudonymize_columns,
    add_laplace_noise,
    generalize_numeric_to_ranges,
    generalize_categorical_by_mapping,
    shuffle_columns,
)

# ============================================================
# 1. Create sample sensitive data
# ============================================================

np.random.seed(42)
data = pd.DataFrame(
    {
        "patient_id": [f"P{i:03d}" for i in range(10)],
        "name": [
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Eve",
            "Frank",
            "Grace",
            "Henry",
            "Ivy",
            "Jack",
        ],
        "age": np.random.randint(20, 80, 10),
        "salary": np.random.randint(30000, 120000, 10),
        "department": np.random.choice(["HR", "IT", "Sales", "Finance"], 10),
        "diagnosis": [
            "A01",
            "B02",
            "A01",
            "C03",
            "B02",
            "A01",
            "D04",
            "B02",
            "C03",
            "A01",
        ],
    }
)

print("Original data:")
print(data)

# ============================================================
# 2. Pseudonymization - Replace identifiers with hashes
# ============================================================

data_pseudo = pseudonymize_columns(
    data.copy(),
    columns=["patient_id", "name"],
    salt="my_secret_salt",  # Recommended for security
)

print("\n--- Pseudonymized Data ---")
print(data_pseudo[["patient_id", "name"]].head())

# ============================================================
# 3. Laplace Noise - Differential Privacy
# ============================================================

data_noisy = add_laplace_noise(
    data.copy(),
    columns=["age", "salary"],
    epsilon=1.0,  # Privacy budget (smaller = more privacy)
)

print("\n--- Laplace Noise (ε=1.0) ---")
print("Original ages:", data["age"].values[:5])
print("Noisy ages:   ", data_noisy["age"].values[:5])

# ============================================================
# 4. Generalization - Numeric to Ranges
# ============================================================

# Convert exact ages to ranges (k-anonymity style)
data_gen_numeric = generalize_numeric_to_ranges(
    data.copy(),
    columns=["age"],  # List of columns
    num_bins=4,  # Number of bins
)

print("\n--- Generalized Age Ranges ---")
print(data_gen_numeric[["age"]].value_counts())

# ============================================================
# 5. Generalization - Categorical Mapping
# ============================================================

# Generalize department to broader categories
dept_mapping = {
    "HR": "Support",
    "IT": "Technical",
    "Sales": "Business",
    "Finance": "Support",
}

data_gen_cat = generalize_categorical_by_mapping(
    data.copy(), column="department", mapping=dept_mapping
)

print("\n--- Generalized Departments ---")
print(data_gen_cat["department"].value_counts())

# ============================================================
# 6. Column Shuffling - Break Correlations
# ============================================================

# Shuffle sensitive columns to break record linkage
data_shuffled = shuffle_columns(data.copy(), columns=["salary", "diagnosis"], seed=42)

print("\n--- Shuffled Columns ---")
print("Original correlations preserved within shuffled columns")
print("but broken between rows")

# ============================================================
# 7. Combined Privacy Pipeline
# ============================================================


def apply_privacy_pipeline(df):
    """Apply multiple privacy transformations."""
    df = pseudonymize_columns(df, ["patient_id", "name"])
    df = add_laplace_noise(df, ["salary"], epsilon=0.5)
    df = generalize_numeric_to_ranges(
        df, "age", bins=[0, 40, 60, 100], labels=["Young", "Middle", "Senior"]
    )
    return df


data_private = apply_privacy_pipeline(data.copy())
print("\n--- Full Privacy Pipeline Applied ---")
print(data_private.head())

print("\n✅ Privacy tutorial completed!")
