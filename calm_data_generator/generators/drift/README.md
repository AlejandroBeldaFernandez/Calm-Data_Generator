# `DriftInjector` Documentation

The `DriftInjector` is a flexible tool designed to inject various types of data drift into existing pandas DataFrames. It supports feature drift, label drift, and concept drift, with customizable profiles like abrupt, gradual, and incremental changes.

## Installation

The `DriftInjector` is part of the `calmops` package. Ensure `calmops` is installed.

## Basic Usage

```python
from calmops.data_generators.DriftInjection.DriftInjector import DriftInjector
import pandas as pd
import numpy as np

# Create a sample dataframe
df = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
    'target': np.random.randint(0, 2, 1000)
})

# Initialize injector
injector = DriftInjector(df)

# Inject drift
drifted_df = injector.add_drift(
    drift_type='feature_shift',
    column='feature1',
    magnitude=2.0,
    start_idx=500,
    end_idx=1000
)
```

## Drift Types

### Feature Drift
- **Shift:** Adds a constant value to the feature.
- **Scale:** Multiplies the feature by a factor.
- **Noise:** Adds Gaussian noise.

### Label Drift
- **Flip:** Randomly flips labels for classification tasks.

### Concept Drift
- **Target Shift:** Changes the distribution of the target variable.

## Drift Profiles

- **Abrupt:** Sudden change at a specific point.
- **Gradual:** Change happens over a transition period.
- **Incremental:** Change increases linearly over time.


## API Reference: Drift Methods

This section lists the available drift injection methods you can use in the `drift_injection_config` list.

### 1. Feature Drift (`inject_feature_drift` family)

Modifies the values of feature columns.

| Method Name | Key Parameters | Description |
| :--- | :--- | :--- |
| `inject_feature_drift` | `feature_cols`, `drift_type` (gaussian_noise, shift, scale), `drift_magnitude` | Base method. Applies drift to a selection. |
| `inject_feature_drift_gradual` | Same as above + `center`, `width` | Applies drift gradually using a sigmoid window. |
| `inject_feature_drift_abrupt` | Same as above + `change_index` | Applies drift suddenly at a specific point. |
| `inject_feature_drift_incremental` | Same as above | Drift increases linearly over the selection. |
| `inject_conditional_drift` | `conditions` (e.g. `[{"column": "age", "operator": ">", "value": 50}]`) | Applies drift only to rows matching conditions. |
| `inject_missing_values_drift` | `missing_fraction` | Sets a fraction of values to `NaN`. |

### 2. Label Drift (`inject_label_drift` family)

Modifies the target/label column.

| Method Name | Key Parameters | Description |
| :--- | :--- | :--- |
| `inject_label_drift` | `target_cols`, `drift_magnitude` (flip rate) | Randomly flips labels. |
| `inject_label_drift_gradual` | Same as above + `center`, `width` | Flip rate increases gradually. |
| `inject_label_drift_abrupt` | Same as above + `change_index` | Flip rate changes suddenly. |

### 3. Advanced / Other

| Method Name | Key Parameters | Description |
| :--- | :--- | :--- |
| `inject_new_category_drift` | `feature_col`, `new_category` | Introduces a previously unseen category. |
| `inject_correlation_matrix_drift` | `target_correlation_matrix` | Alters the relationships between features. |

### Configuration Example

When using with a Generator (e.g., `RealGenerator`), use these method names in your config:

```python
drift_config = [
    {
        "method": "inject_feature_drift",
        "params": {
            "feature_cols": ["age"],
            "drift_type": "shift",
            "drift_magnitude": 10.0
        }
    },
    {
        "method": "inject_missing_values_drift",
        "params": {
            "feature_cols": ["income"],
            "missing_fraction": 0.2
        }
    }
]
```

## Tutorial

For a complete example, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```

