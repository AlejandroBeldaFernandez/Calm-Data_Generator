# `RealGenerator` Documentation

The `Real` generator module provides tools to synthesize data that mimics the statistical properties of real-world datasets. It leverages the `SDV` (Synthetic Data Vault) library for advanced synthesis and includes block-based generation capabilities.

## Generators

### 1. `RealGenerator`
A wrapper around SDV models to learn from a real dataset and generate synthetic samples.

### 2. `RealBlockGenerator`
Extends `RealGenerator` to handle block-structured data, allowing for:
- Independent generation per block.
- Dynamic block creation based on size or time.
- Drift scheduling across blocks.

## Installation

Ensure `calmops` and `sdv` are installed.

## Basic Usage (`RealGenerator`)

```python
from calmops.data_generators.Real.RealGenerator import RealGenerator
import pandas as pd

# Load real data
real_data = pd.read_csv("my_real_data.csv")

# Initialize (auto-fits)
generator = RealGenerator(original_data=real_data, method="cart")

# Generate synthetic data
synthetic_data = generator.generate(n_samples=1000)
```

## Block Usage (`RealBlockGenerator`)

```python
from calmops.data_generators.Real.RealBlockGenerator import RealBlockGenerator

block_gen = RealBlockGenerator(model_name="GaussianCopula")
block_gen.fit(real_data)

# Generate blocks
blocks = block_gen.generate_blocks(
    n_blocks=5,
    block_size=200,
    drift_schedule={2: {'drift_type': 'feature_shift', 'magnitude': 1.5}}
)
```

## Tutorial

For a complete example, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```
