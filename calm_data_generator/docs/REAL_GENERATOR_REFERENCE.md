# RealGenerator - Complete Reference

**Location:** `calm_data_generator.generators.tabular.RealGenerator`

The main generator for tabular data synthesis from real datasets.

---

## Initialization

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator(
    auto_report=True,       # Automatically generate report after synthesis
    minimal_report=False,   # If True, faster report without correlations/PCA
    random_state=42,        # Seed for reproducibility
    logger=None,            # Optional custom Python logger
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_report` | bool | `True` | Automatically generate a quality report |
| `minimal_report` | bool | `False` | Simplified report (faster) |
| `random_state` | int | `None` | Seed for reproducibility |
| `logger` | Logger | `None` | Custom Python Logger instance |

---

## Main Method: `generate()`

```python
synthetic_df = gen.generate(
    data=df,                          # Original DataFrame (required)
    n_samples=1000,                   # Number of samples to generate (required)
    method="ctgan",                   # Synthesis method
    target_col="target",              # Target column (optional)
    output_dir="./output",            # Output directory
    generator_name="my_generator",    # Base name for output files
    save_dataset=False,               # Save resulting CSV
    # Model parameters
    model_params={...},               # Method-specific parameters
    # Custom distributions
    custom_distributions={"target": {0: 0.3, 1: 0.7}},
    # Date injection
    date_col="date",
    date_start="2024-01-01",
    date_step={"days": 1},
    # Post-processing
    drift_injection_config=[...],
    dynamics_config={...},
    constraints=[...],
)
```

### `generate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Original dataset (required) |
| `n_samples` | int | - | Number of samples to generate (required) |
| `method` | str | `"cart"` | Synthesis method |
| `target_col` | str | `None` | Target column for balancing |
| `output_dir` | str | `None` | Directory for output files |
| `generator_name` | str | `"RealGenerator"` | Base name for output files |
| `save_dataset` | bool | `False` | Save generated dataset as CSV |
| `custom_distributions` | Dict | `None` | Forced distribution per column |
| `date_col` | str | `None` | Name of date column to inject |
| `date_start` | str | `None` | Start date ("YYYY-MM-DD") |
| `date_step` | Dict | `None` | Time increment (e.g., `{"days": 1}`) |
| `date_every` | int | `1` | Increment date every N rows |
| `drift_injection_config` | List[Dict] | `None` | Post-generation drift configuration |
| `dynamics_config` | Dict | `None` | Dynamic evolution configuration |
| `model_params` | Dict | `None` | Specific model parameters |
| `constraints` | List[Dict] | `None` | Integrity constraints |

---

## Full `model_params` Reference

The `model_params` dictionary allows fine-tuning internal parameters for each synthesis method.

### Deep Learning (SDV)

| Parameter | Default | Methods | Description |
|-----------|---------|---------|-------------|
| `sdv_epochs` | 300 | ctgan, tvae, copula | Number of training epochs |
| `sdv_batch_size` | 100 | ctgan, tvae, copula | Training batch size |

**Example:**
```python
model_params={"sdv_epochs": 500, "sdv_batch_size": 256}
```

### CART (Decision Trees)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cart_iterations` | 10 | Number of FCS iterations |
| `cart_min_samples_leaf` | None | Minimum samples per leaf (auto if None) |

**Example:**
```python
model_params={"cart_iterations": 20, "cart_min_samples_leaf": 5}
```

### Random Forest

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rf_n_estimators` | None | Number of trees (auto if None) |
| `rf_min_samples_leaf` | None | Minimum samples per leaf |

**Example:**
```python
model_params={"rf_n_estimators": 100, "rf_min_samples_leaf": 3}
```

### LightGBM

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lgbm_n_estimators` | None | Number of trees |
| `lgbm_learning_rate` | None | Learning rate |

**Example:**
```python
model_params={"lgbm_n_estimators": 200, "lgbm_learning_rate": 0.05}
```

### Gaussian Mixture Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gmm_n_components` | 5 | Number of Gaussian components |
| `gmm_covariance_type` | "full" | Covariance type: "full", "tied", "diag", "spherical" |

**Example:**
```python
model_params={"gmm_n_components": 10, "gmm_covariance_type": "diag"}
```

### SMOTE / ADASYN

| Parameter | Default | Methods | Description |
|-----------|---------|---------|-------------|
| `smote_neighbors` | 5 | smote | Number of k-NN neighbors |
| `adasyn_neighbors` | 5 | adasyn | Number of k-NN neighbors |

**Example:**
```python
model_params={"smote_neighbors": 7}
```

### Differential Privacy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dp_epsilon` | 1.0 | Privacy ε parameter (lower = more private) |
| `dp_delta` | 1e-5 | Privacy δ parameter |

**Example:**
```python
model_params={"dp_epsilon": 0.5, "dp_delta": 1e-6}
```

### DataSynthesizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ds_k` | 5 | Bayesian network degree (dependency complexity) |

**Example:**
```python
model_params={"ds_k": 3}  # k=1 is independent, higher k captures more correlations
```

### Time Series (PAR)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `par_epochs` | 100 | PAR training epochs |
| `sequence_index` | None | Column identifying each sequence/entity |

**Example:**
```python
model_params={"par_epochs": 200, "sequence_index": "patient_id"}
```

### Diffusion Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diffusion_steps` | 50 | Number of diffusion steps (higher = better quality, slower) |

**Example:**
```python
model_params={"diffusion_steps": 100}
```

---

## Available Synthesis Methods

### Deep Learning

| Method | Description | Key Parameters | Dependencies |
|--------|-------------|----------------|--------------|
| `ctgan` | Conditional Tabular GAN | `epochs`, `batch_size` | SDV |
| `tvae` | Tabular Variational Autoencoder | `epochs`, `batch_size` | SDV |
| `copulagan` | Copula-based GAN | `epochs`, `batch_size` | SDV |
| `diffusion` | Tabular Diffusion Models | `steps` | PyTorch |

### Statistical Models

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `gaussian_copula` | Gaussian Copula | - |
| `gmm` | Gaussian Mixture Model | `gmm_n_components`, `gmm_covariance_type` |

### Fully Conditional Specification (FCS)

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `cart` | Decision Trees | `min_samples_leaf`, `iterations` |
| `rf` | Random Forest | `n_estimators`, `min_samples_leaf`, `iterations` |
| `lgbm` | LightGBM | `n_estimators`, `learning_rate`, `iterations` |

### Oversampling

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `smote` | Classic SMOTE | `n_neighbors` |
| `adasyn` | Adaptive SMOTE | `n_neighbors` |
| `resample` | Simple Bootstrap | - |

### Privacy

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `dp` | Differential Privacy | `epsilon`, `delta` |
| `datasynth` | DataSynthesizer | `k` |

### Time Series

| Method | Description | Key Parameters | Dependencies |
|--------|-------------|----------------|--------------|
| `par` | Probabilistic AutoRegressive | `epochs`, `sequence_key` | SDV |
| `timegan` | TimeGAN | `epochs`, `seq_len`, `sequence_key` | ydata-synthetic |
| `dgan` | DoppelGANger | `epochs`, `seq_len`, `sequence_key` | ydata-synthetic |
| `copula_temporal` | Temporal Copula | `sequence_key`, `time_col` | SDV |

---

## Advanced Features

### Custom Distributions

```python
synthetic = gen.generate(
    data=df,
    n_samples=5000,
    method="ctgan",
    target_col="Churn",
    custom_distributions={
        "Churn": {0: 0.5, 1: 0.5},
        "Region": {"North": 0.4, "South": 0.6},
    }
)
```

### Date Injection

```python
synthetic = gen.generate(
    data=df,
    n_samples=1000,
    method="cart",
    date_col="timestamp",
    date_start="2024-06-01",
    date_step={"hours": 1},
    date_every=1,
)
```

### Post-Generation Drift

```python
synthetic = gen.generate(
    data=df,
    n_samples=1000,
    method="tvae",
    drift_injection_config=[
        {
            "method": "inject_feature_drift",
            "feature_cols": ["price"],
            "drift_type": "shift",
            "drift_magnitude": 0.3,
            "start_index": 500,
        }
    ],
)
```

---

## Automatic Reports

When `auto_report=True`, the following are generated:

- `quality_report.html`: Data profiling report
- `comparison_report.html`: Real vs. Synthetic comparison
- `sdv_quality_report.json`: Detailed SDV metrics
- `distribution_plots.png`: Distribution visualizations
- `correlation_heatmap.png`: Correlation maps

---

## Best Practices

6. **Severe imbalance:** Use `smote` or `adasyn` with `target_col`.

---

## Comprehensive Use Cases

### Case 1: Fraud Detection (Imbalanced Data)

**Scenario:** You have a dataset of credit card transactions where only 0.1% are fraudulent. You want to train a model but need more fraud cases.

**Solution:** Use `smote` or `ctgan` with forced distribution.

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()

# Option A: SMOTE (Oversampling)
synthetic_fraud = gen.generate(
    data=df,
    n_samples=5000,
    method="smote",
    target_col="is_fraud",
    model_params={"smote_neighbors": 5}
)

# Option B: CTGAN with Controlled Sampling
synthetic_balanced = gen.generate(
    data=df,
    n_samples=5000,
    method="ctgan",
    target_col="is_fraud",
    custom_distributions={"is_fraud": {0: 0.5, 1: 0.5}}, # Force 50/50 split
    model_params={"sdv_epochs": 500}
)
```

### Case 2: Multi-Entity Time Series (Patients)

**Scenario:** You have longitudinal data for multiple patients. Each patient has multiple rows (visits) ordered by time. You want to generate new synthetic patients with realistic visit trajectories.

**Solution:** Use `par` (Probabilistic AutoRegressive) model.

```python
# Data structure: [patient_id, visit_date, heart_rate, blood_pressure]

synthetic_patients = gen.generate(
    data=patient_df,
    n_samples=100,  # Generate 100 new patients (sequences)
    method="par",
    model_params={
        "sequence_index": "patient_id",  # Column identifying entities
        "par_epochs": 150
    }
)
```

### Case 3: Privacy-Preserving Dataset Sharing

**Scenario:** You need to share a customer dataset with an external partner but must guarantee differential privacy to comply with GDPR.

**Solution:** Use `dp` (Differential Privacy) with PATE-CTGAN algorithm.

```python
synthetic_safe = gen.generate(
    data=customer_df,
    n_samples=10000,
    method="dp",
    model_params={
        "dp_epsilon": 0.5,  # Strict privacy budget (lower is more private)
        "dp_delta": 1e-5
    }
)
```

### Case 4: High-Performance Upsampling for Large Datasets

**Scenario:** You have a 1M row dataset and need 5M rows for stress testing a database. Deep learning methods are too slow.

**Solution:** Use `lgbm` (LightGBM) or `rf` (Random Forest) which are faster and scalable.

```python
synthetic_large = gen.generate(
    data=large_df,
    n_samples=5_000_000,
    method="lgbm",
    model_params={
        "lgbm_n_estimators": 100,
        "lgbm_learning_rate": 0.1
    },
    minimal_report=True  # Skip heavy reporting to save time
)
```
