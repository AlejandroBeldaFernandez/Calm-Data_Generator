# CALM-Data-Generator - Full Documentation

## Table of Contents

1. [Installation](#installation)
2. [Module Overview](#module-overview)
3. [RealGenerator](#realgenerator)
4. [ClinicalDataGenerator](#clinicgenerator)
5. [StreamGenerator](#syntheticgenerator)
6. [DriftInjector](#driftinjector)
7. [ScenarioInjector](#scenarioinjector)
8. [Privacy Module](#privacy-module)
9. [Configuration Options](#configuration-options)
10. [Best Practices](#best-practices)

---

## Installation

### Basic Installation
```bash
pip install calm_data_generator
```

### Optional Dependencies

| Extra | Command | Includes |
|-------|---------|----------|
| deeplearning | `pip install calm_data_generator[deeplearning]` | SDV, PyTorch, smartnoise-synth, ydata-synthetic |
| profiling | `pip install calm_data_generator[profiling]` | ydata-profiling, sweetviz |
| full | `pip install calm_data_generator[full]` | All optional dependencies |

---

## Module Overview

```
calm_data_generator/
├── generators/
│   ├── tabular/    → RealGenerator, QualityReporter
│   ├── clinical/   → ClinicalDataGenerator
│   ├── stream/     → StreamGenerator
│   ├── drift/      → DriftInjector
│   └── dynamics/   → ScenarioInjector
├── anonymizer/     → Privacy transformations
└── reports/        → Visualization & reporting
```

---

## RealGenerator

Generates synthetic data from existing real datasets.

### Basic Usage

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()

# Generate with default method (CART)
synthetic = gen.generate(
    data=df,
    n_samples=1000
)
```

### Available Methods

#### Machine Learning Methods

```python
# CART (Classification and Regression Trees)
synthetic = gen.generate(df, 1000, method='cart')

# Random Forest
synthetic = gen.generate(df, 1000, method='rf')

# LightGBM
synthetic = gen.generate(df, 1000, method='lgbm')

# Gaussian Mixture Model
synthetic = gen.generate(df, 1000, method='gmm')
```

#### Deep Learning Methods

```python
# CTGAN - Conditional GAN for Tabular Data
synthetic = gen.generate(
    df, 1000, 
    method='ctgan',
    model_params={'sdv_epochs': 300}
)

# TVAE - Variational Autoencoder
synthetic = gen.generate(
    df, 1000,
    method='tvae',
    model_params={'sdv_epochs': 300}
)

# Gaussian Copula
synthetic = gen.generate(df, 1000, method='copula')

# Tabular Diffusion (DDPM)
synthetic = gen.generate(
    df, 1000,
    method='diffusion',
    model_params={'diffusion_steps': 50}
)
```

#### Augmentation Methods

```python
# SMOTE Oversampling
synthetic = gen.generate(
    df, 1000,
    method='smote',
    target_col='label'
)

# ADASYN Adaptive Sampling
synthetic = gen.generate(
    df, 1000,
    method='adasyn',
    target_col='label'
)
```

#### Privacy-Preserving Methods

```python
# Differential Privacy (PATE-CTGAN)
synthetic = gen.generate(
    df, 1000,
    method='dp',
    model_params={
        'dp_epsilon': 1.0,
        'dp_delta': 1e-5
    }
)
```

#### Time Series Methods

```python
# PAR (Probabilistic AutoRegressive)
synthetic = gen.generate(
    df, 100,
    method='par',
    block_column='entity_id',
    model_params={'par_epochs': 100}
)

# TimeGAN
synthetic = gen.generate(
    df, 100,
    method='timegan',
    block_column='entity_id',
    model_params={
        'seq_len': 24,
        'timegan_epochs': 100
    }
)

# DGAN (DoppelGANger)
synthetic = gen.generate(
    df, 100,
    method='dgan',
    block_column='entity_id',
    model_params={'seq_len': 24}
)

# Temporal Copula
synthetic = gen.generate(
    df, 100,
    method='copula_temporal',
    block_column='entity_id',
    model_params={'time_col': 'timestamp'}
)
```

### Constraints

Apply post-hoc filtering with business rules:

```python
synthetic = gen.generate(
    df, 1000,
    method='cart',
    constraints=[
        {'col': 'age', 'op': '>=', 'val': 18},
        {'col': 'age', 'op': '<=', 'val': 100},
        {'col': 'income', 'op': '>', 'val': 0},
        {'col': 'status', 'op': '==', 'val': 'active'}
    ]
)
```

**Supported Operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`

---

## ClinicalDataGenerator

Generates realistic clinical/medical datasets.

### Basic Generation

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator, DateConfig

gen = ClinicalDataGenerator()

result = gen.generate(
    n_samples=100,
    n_genes=500,
    n_proteins=200,
    date_config=DateConfig(start_date="2024-01-01")
)

# Access generated data
demographics = result['demographics']
genes = result['genes']
proteins = result['proteins']
```

### Longitudinal Data (Multi-Visit)

```python
result = gen.generate_longitudinal_data(
    n_samples=50,
    longitudinal_config={
        'n_visits': 4,           # 4 visits per patient
        'time_step_days': 90,    # 3 months between visits
        'evolution_config': {
            'features': ['Age', 'Propensity'],
            'trend': 0.02,       # 2% change per visit
            'noise': 0.01        # Random noise
        }
    },
    date_config=DateConfig(start_date="2024-01-01")
)

longitudinal = result['longitudinal']
```

### Clinical Constraints

```python
result = gen.generate(
    n_samples=100,
    constraints=[
        {'col': 'Age', 'op': '>=', 'val': 18},
        {'col': 'Age', 'op': '<=', 'val': 85}
    ]
)
```

---

## StreamGenerator

Stream-based data generation compatible with River library.

### With Python Generator

```python
from calm_data_generator.generators.stream import StreamGenerator

def my_stream():
    while True:
        x = {'f1': random(), 'f2': random()}
        y = 1 if x['f1'] > 0.5 else 0
        yield x, y

gen = StreamGenerator()
synthetic = gen.generate(
    generator_instance=my_stream(),
    n_samples=1000
)
```

### With River Generators

```python
from river import synth

# Use River's built-in generators
stream = synth.Agrawal(seed=42)

synthetic = gen.generate(
    generator_instance=stream,
    n_samples=1000
)
```

### Balanced Generation

```python
# Balance classes
synthetic = gen.generate(
    generator_instance=stream,
    n_samples=1000,
    balance_target=True,
    use_smote=True  # Optional SMOTE
)
```

### Sequence Generation

```python
synthetic = gen.generate(
    generator_instance=stream,
    n_samples=1000,
    date_start='2024-01-01',
    sequence_config={
        'entity_col': 'user_id',
        'events_per_entity': 10
    }
)
```

---

## DriftInjector

Inject controlled drift into datasets for ML testing.

### Gradual Feature Drift

```python
from calm_data_generator.generators.drift import DriftInjector

injector = DriftInjector(time_col='timestamp')

# Gradual drift with smooth transition window
drifted = injector.inject_feature_drift_gradual(
    df=data,
    feature_cols=['feature1', 'feature2'],
    drift_magnitude=0.5,
    drift_type='shift',          # gaussian_noise, shift, scale
    start_index=50,
    center=25,                   # Center of transition
    width=20,                    # Width of transition
    profile='sigmoid',           # sigmoid, linear, cosine
    auto_report=False
)
```

### Abrupt Feature Drift

```python
# Immediate drift from a specific index
drifted = injector.inject_feature_drift(
    df=data,
    feature_cols=['feature1'],
    drift_magnitude=0.8,
    drift_type='shift',
    start_index=60,
    auto_report=False
)
```

### Drift Types

| Type | Description |
|------|-------------|
| `gaussian_noise` | Add Gaussian noise scaled by magnitude |
| `shift` | Shift values by magnitude × mean |
| `scale` | Scale values by 1 + magnitude |
| `add_value` | Add specific value (requires `drift_value`) |
| `subtract_value` | Subtract specific value |
| `multiply_value` | Multiply by specific value |

### Label Drift

```python
# Gradual label flips
drifted = injector.inject_label_drift_gradual(
    df=data,
    target_col='label',
    drift_magnitude=0.3,     # 30% flip probability
    start_index=70,
    auto_report=False
)
```

### Conditional Drift

```python
# Apply drift only to rows meeting conditions
drifted = injector.inject_conditional_drift(
    df=data,
    feature_cols=['feature2'],
    conditions=[
        {'column': 'age', 'operator': '>', 'value': 50}
    ],
    drift_type='shift',
    drift_magnitude=0.5,
    auto_report=False
)
```

### Outlier Injection

```python
drifted = injector.inject_outliers_global(
    df=data,
    cols=['feature1', 'feature2'],
    outlier_prob=0.05,       # 5% of rows
    factor=3.0,              # Outlier magnitude
    auto_report=False
)
```

### Label Shift (Distribution Change)

```python
# Change label distribution to 30% class 0, 70% class 1
drifted = injector.inject_label_shift(
    df=data,
    target_col='label',
    target_distribution={0: 0.3, 1: 0.7},
    auto_report=False
)
```

### Correlation Matrix Drift

```python
import numpy as np

# Define target correlation structure
target_corr = np.array([
    [1.0, 0.8, 0.2],
    [0.8, 1.0, 0.5],
    [0.2, 0.5, 1.0]
])

drifted = injector.inject_correlation_matrix_drift(
    df=data,
    feature_cols=['f1', 'f2', 'f3'],
    target_correlation_matrix=target_corr,
    auto_report=False
)
```

### New Category Drift

```python
# Introduce a new category "D" that didn't exist before
drifted = injector.inject_new_category_drift(
    df=data,
    feature_col='category',
    new_category='D',
    probability=0.15,        # 15% of rows get new category
    replace_categories=['A', 'B'],  # Only replace A or B
    auto_report=False
)
```

---

## ScenarioInjector

Feature evolution and target variable construction.

### Feature Evolution

```python
from calm_data_generator.generators.dynamics import ScenarioInjector

injector = ScenarioInjector(seed=42)

evolution_config = {
    'temperature': {
        'type': 'trend',
        'rate': 0.05,
        'noise_std': 0.5
    },
    'humidity': {
        'type': 'cyclic',
        'period': 30,
        'amplitude': 5
    }
}

evolved = injector.evolve_features(
    df=data,
    evolution_config=evolution_config,
    time_col='timestamp'
)
```

### Target Construction

```python
# Regression target
data = injector.construct_target(
    df=data,
    target_col='consumption',
    formula='temperature * 2.5 + humidity * 0.8',
    noise_std=5.0,
    task_type='regression'
)

# Classification target
data = injector.construct_target(
    df=data,
    target_col='is_high',
    formula='value1 + value2',
    task_type='classification',
    threshold=50
)
```

### Future Projection

```python
future = injector.project_to_future_period(
    df=data,
    periods=3,
    period_length=30,
    trend_config={
        'temperature': 0.02,
        'humidity': -0.01
    },
    time_col='timestamp'
)
```

---

## Anonymizer (Privacy Module)

### Pseudonymization

```python
from calm_data_generator.anonymizer import pseudonymize_columns

data = pseudonymize_columns(
    df,
    columns=['patient_id', 'name'],
    salt='optional_salt_string'  # Recommended for security
)
```

### Differential Privacy (Laplace Noise)

```python
from calm_data_generator.anonymizer import add_laplace_noise

data = add_laplace_noise(
    df,
    columns=['age', 'salary'],
    epsilon=1.0  # Privacy budget (smaller = more privacy)
)
```

### Generalization

```python
from calm_data_generator.anonymizer import generalize_numeric_to_ranges

data = generalize_numeric_to_ranges(
    df,
    columns=['age'],       # List of columns
    num_bins=5             # Number of bins/ranges to create
)
```

### Shuffling

```python
from calm_data_generator.anonymizer import shuffle_columns

data = shuffle_columns(
    df,
    columns=['salary', 'diagnosis'],
    random_state=42
)
```

---

## Configuration Options

### DateConfig

```python
from calm_data_generator.generators.configs import DateConfig

config = DateConfig(
    start_date="2024-01-01",
    date_col="timestamp",
    date_every=1,  # Every N rows
    date_step={'days': 1}
)
```

### Model Parameters

```python
model_params = {
    # SDV methods
    'sdv_epochs': 300,
    'batch_size': 500,
    
    # Diffusion
    'diffusion_steps': 50,
    
    # Time Series
    'par_epochs': 100,
    'timegan_epochs': 100,
    'seq_len': 24,
    
    # Privacy
    'dp_epsilon': 1.0,
    'dp_delta': 1e-5,
    
    # Augmentation
    'smote_neighbors': 5,
    'adasyn_neighbors': 5
}
```

---

## Best Practices

### 1. Start with Fast Methods
```python
# Try CART first for quick iteration
synthetic = gen.generate(df, 1000, method='cart')
```

### 2. Validate Synthetic Data
```python
from calm_data_generator.generators.tabular import QualityReporter

reporter = QualityReporter()
report = reporter.generate_report(original_df, synthetic_df)
```

### 3. Use Constraints for Business Rules
```python
# Always validate domain constraints
constraints = [
    {'col': 'probability', 'op': '>=', 'val': 0},
    {'col': 'probability', 'op': '<=', 'val': 1}
]
```

### 4. Privacy First for Sensitive Data
```python
# Apply privacy transformations before sharing
from calm_data_generator.privacy import pseudonymize_columns, add_laplace_noise

private_data = pseudonymize_columns(df, ['id', 'name'])
private_data = add_laplace_noise(private_data, ['salary'], epsilon=1.0)
```

---

## CLI Commands

```bash
# List tutorials
calm_data_generator tutorials

# View tutorial
calm_data_generator tutorials show 1

# Run tutorial
calm_data_generator tutorials run 1

# Show paths
calm_data_generator tutorials path

# Show version
calm_data_generator version

# Access docs
calm_data_generator docs
```

---

## Support

For issues and questions:
- GitHub Issues: [https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator/issues](https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator/issues)
- Email: alejandrobeldafernandez@gmail.com
