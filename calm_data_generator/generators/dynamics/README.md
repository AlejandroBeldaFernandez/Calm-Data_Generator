# `DynamicsInjector` Documentation

The `DynamicsInjector` is a standalone module designed to evolve features over time and construct target variables based on complex rules. It is ideal for creating scenarios where the underlying data generation process changes dynamically.

## Installation

The `DynamicsInjector` is part of the `calmops` package. Ensure `calmops` is installed.

## Basic Usage

```python
from calmops.data_generators.Dynamics.DynamicsInjector import DynamicsInjector
import pandas as pd
import numpy as np

# Create a sample dataframe
df = pd.DataFrame({
    'time': range(100),
    'feature_A': np.random.normal(0, 1, 100)
})

# Initialize injector
injector = DynamicsInjector(seed=42)

# 1. Evolve Features
evolution_config = {
    'feature_A': {'type': 'linear', 'slope': 0.1}
}
df_evolved = injector.evolve_features(df, evolution_config, time_col='time')

# 2. Construct Target
# Target = 1 if (feature_A > 0.5) else 0
df_target = injector.construct_target(
    df_evolved,
    target_col='target',
    formula='feature_A',
    task_type='classification',
    threshold=0.5
)
```

## Features

### Feature Evolution
Modify existing features based on time `t`.
- **Linear:** `slope * t + intercept`
- **Cycle/Sinusoidal:** `amplitude * sin(2*pi*t/period + phase)`
- **Sigmoid:** `amplitude * sigmoid((t - center)/width)`

### Target Construction
Create a new target variable from existing features.
- **Formula:** String expression (e.g., `"0.5 * Age + Income"`) or callable function.
- **Noise:** Add Gaussian noise to the raw score.
- **Task Type:**
    - `regression`: Returns the raw score.
    - `classification`: Returns binary labels based on a threshold.

## Tutorial

For a complete example, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```
