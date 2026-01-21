# CalmGenerator API Documentation

## Modules Overview

### generators.tabular - Real Data Synthesis

```python
from calm_data_generator.generators.tabular import RealGenerator, QualityReporter
```

**RealGenerator** - Generate synthetic data from real datasets

| Method | Description |
|--------|-------------|
| `cart` | CART-based iterative synthesis |
| `rf` | Random Forest synthesis |
| `lgbm` | LightGBM synthesis |
| `ctgan` | CTGAN (deep learning) |
| `tvae` | TVAE (variational autoencoder) |
| `copula` | Gaussian Copula |
| `smote` | SMOTE oversampling |
| `adasyn` | ADASYN adaptive sampling |
| `dp` | Differential Privacy (PATE-CTGAN) |
| `par` | PAR time series |
| `timegan` | TimeGAN (ydata-synthetic) |
| `dgan` | DoppelGANger (ydata-synthetic) |
| `copula_temporal` | Temporal Copula |
| `diffusion` | Tabular Diffusion (DDPM) |

---

### generators.clinical - Clinical Data

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator, DateConfig
```

**Methods:**
- `generate()` - Generate demographics + omics
- `generate_longitudinal_data()` - Multi-visit patient data

---

### generators.stream - Stream-Based

```python
from calm_data_generator.generators.stream import StreamGenerator
```

**Features:**
- River library compatible
- Balanced generation
- SMOTE post-hoc
- Sequence generation

---

### generators.drift - Drift Injection

```python
from calm_data_generator.generators.drift import DriftInjector
```

**Drift Types:**
- `inject_feature_drift_gradual()`
- `inject_feature_drift_abrupt()`
- `inject_feature_drift_recurrent()`
- `inject_label_drift_gradual()`
- `inject_label_drift_abrupt()`
- `inject_label_drift_incremental()`
- `inject_concept_drift()`
- `inject_conditional_drift()`
- `inject_outliers_global()`
- `inject_new_category_drift()`
- `inject_correlation_matrix_drift()`
- `inject_binary_probabilistic_drift()`
- `inject_multiple_types_of_drift()`

---

### generators.dynamics - Scenario Evolution

```python
from calm_data_generator.generators.dynamics import ScenarioInjector
```

**Methods:**
- `evolve_features()` - Apply trends/cycles
- `construct_target()` - Create target variables
- `project_to_future_period()` - Future data

---

### privacy - Privacy Transformations

```python
from calm_data_generator.privacy import (
    pseudonymize_columns,
    add_laplace_noise,
    generalize_numeric_to_ranges,
    generalize_categorical_by_mapping,
    shuffle_columns
)
```

---

## Installation

```bash
# Basic
pip install calm_data_generator

# Stream (River)
pip install calm_data_generator[stream]

# Time Series (Gretel)
pip install calm_data_generator[timeseries]

# Full
pip install calm_data_generator[full]
```
