# Privacy Module

The `Privacy` module provides a set of tools to apply various privacy-preserving techniques to your datasets. These techniques help in anonymizing data, reducing re-identifiability, and adding differential privacy noise.

## Features

- **Pseudonymization**: Hash sensitive columns (e.g., IDs, names) using SHA256 with an optional salt.
- **Differential Privacy**: Add Laplace noise to numeric columns to protect individual values.
- **Generalization**:
    - **Numeric**: Bin numeric values into ranges (e.g., Age 20-30).
    - **Categorical**: Map specific categories to broader groups (e.g., "New York" -> "USA").
- **Shuffling**: Shuffle values within columns to break correlations while preserving column distributions.

## Installation

Ensure `calmops` is installed.

## Basic Usage

```python
from calmops.privacy.privacy import (
    pseudonymize_columns,
    add_laplace_noise,
    generalize_numeric_to_ranges
)
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'user_id': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 35, 45],
    'salary': [50000, 60000, 70000]
})

# 1. Pseudonymize IDs
df_secure = pseudonymize_columns(df, columns=['user_id'], salt='s3cr3t')

# 2. Add Noise to Salary
df_secure = add_laplace_noise(df_secure, columns=['salary'], epsilon=0.5)

# 3. Generalize Age
df_secure = generalize_numeric_to_ranges(df_secure, columns=['age'], num_bins=3)

print(df_secure)
```

## Tutorial

For a complete walkthrough of all features, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```
