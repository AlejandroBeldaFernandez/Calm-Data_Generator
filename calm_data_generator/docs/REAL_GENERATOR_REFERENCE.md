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
    adversarial_validation=True,      # Activate adversarial validation
)
```

### `generate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Original and reference dataset (required) |
| `n_samples` | int | - | Number of samples to generate (required) |
| `method` | str | `"cart"` | Synthesis method to use |
| `target_col` | str | `None` | Target variable column name for balancing |
| `output_dir` | str | `None` | Directory to save reports and datasets |
| `generator_name` | str | `"RealGenerator"` | Base name for output files |
| `save_dataset` | bool | `False` | Whether to save the generated dataset to a CSV file |
| `custom_distributions` | Dict | `None` | Force specific distributions for columns |
| `date_col` | str | `None` | Name of the date column to inject |
| `date_start` | str | `None` | Start date ("YYYY-MM-DD") |
| `date_step` | Dict | `None` | Time increment (e.g., `{"days": 1}`) |
| `date_every` | int | `1` | Increment date every N rows |
| `drift_injection_config` | List[Dict] | `None` | Configuration for post-generation drift injection |
| `dynamics_config` | Dict | `None` | Configuration for dynamic feature evolution |
| `model_params` | Dict | `None` | Model-specific hyperparameters (passed as `**kwargs`) |
| `constraints` | List[Dict] | `None` | Integrity constraints |
| `adversarial_validation` | bool | `False` | Activate Discriminator Report (Real vs Synthetic) |

---

## Full `model_params` Reference

The `model_params` dictionary allows fine-tuning internal parameters for each synthesis method.

### Deep Learning (SDV)

| Parameter | Methods | Description |
|-----------|---------|-------------|
| `epochs` | All SDV | Number of training epochs |
| `batch_size` | All SDV | Training batch size |
| `verbose` | All SDV | Enable detailed training logs |
| `**kwargs` | All | Any parameter supported by the underlying model (e.g., `discriminator_steps` for CTGAN) |

**Example:**
```python
model_params={
    "epochs": 500, 
    "batch_size": 256,
    "discriminator_steps": 5  # Specific to CTGAN
}
```

### CART (Decision Trees)

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `min_samples_leaf` | Minimum samples per leaf (auto if None) |
| `**kwargs` | Any parameter supported by sklearn's DecisionTree |

**Example:**
```python
model_params={"iterations": 20, "min_samples_leaf": 5, "max_depth": 10}
```

### Random Forest

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `n_estimators` | Number of trees |
| `min_samples_leaf` | Minimum samples per leaf |
| `**kwargs` | Any parameter supported by sklearn's RandomForest |

**Example:**
```python
model_params={"n_estimators": 100, "min_samples_leaf": 3, "max_depth": 15}
```

### LightGBM

| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of FCS iterations (default: 10) |
| `n_estimators` | Number of boosting rounds |
| `learning_rate` | Learning rate |
| `**kwargs` | Any parameter supported by LightGBM |

**Example:**
```python
model_params={"n_estimators": 200, "learning_rate": 0.05, "max_depth": 8}
```

### Gaussian Mixture Models

| Parameter | Description |
|-----------|-------------|
| `n_components` | Number of Gaussian components (default: 5) |
| `covariance_type` | Covariance type: "full", "tied", "diag", "spherical" (default: "full") |
| `**kwargs` | Any parameter supported by sklearn's GaussianMixture |

**Example:**
```python
model_params={"n_components": 10, "covariance_type": "diag"}
```

### SMOTE / ADASYN

| Parameter | Description |
|-----------|-------------|
| `k_neighbors` | Number of k-NN neighbors for SMOTE (default: 5) |
| `n_neighbors` | Number of k-NN neighbors for ADASYN (default: 5) |
| `**kwargs` | Any parameter supported by imbalanced-learn's SMOTE/ADASYN |

**Example:**
```python
model_params={"k_neighbors": 7}  # For SMOTE
model_params={"n_neighbors": 5}  # For ADASYN
```

### Differential Privacy

| Parameter | Description |
|-----------|-------------|
| `epsilon` | Privacy ε parameter (default: 1.0, lower = more private) |
| `delta` | Privacy δ parameter (default: 1e-5) |
| `synth_type` | Synthesizer type: "pate_ctgan", "mwem", etc. (default: "pate_ctgan") |

**Example:**
```python
model_params={"epsilon": 0.5, "delta": 1e-6, "synth_type": "pate_ctgan"}
```

### DataSynthesizer

| Parameter | Description |
|-----------|-------------|
| `k` | Bayesian network degree (default: 5, k=1 is independent) |

**Example:**
```python
model_params={"k": 3}  # Higher k captures more correlations
```

### Time Series (PAR)

| Parameter | Description |
|-----------|-------------|
| `epochs` | PAR training epochs (default: 100) |
| `sequence_key` | Column identifying each sequence/entity |
| `**kwargs` | Any parameter supported by SDV's PARSynthesizer |

**Example:**
```python
model_params={"epochs": 200, "sequence_key": "patient_id"}
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

### Single-Cell / High-Dimensional

| Method | Description | Key Parameters | Dependencies |
|--------|-------------|----------------|--------------|
| `scvi` | scVI Variational Autoencoder | `epochs`, `n_latent`, `n_layers` | scvi-tools |
| `scgen` | scGen (perturbation prediction) | `epochs`, `n_latent`, `condition_col` | scvi-tools |

---

## Method Selection Guide

Choose the right method based on your data and requirements:

| Use Case | Recommended Methods | Why |
|----------|---------------------|-----|
| **General tabular data** | `ctgan`, `tvae` | Best balance of quality and speed |
| **Small datasets (< 1000 rows)** | `cart`, `rf`, `gmm` | Don't overfit, fast |
| **Large datasets (> 100k rows)** | `lgbm`, `ctgan` | Scalable |
| **Preserve correlations** | `ctgan`, `copulagan`, `datasynth` | Capture feature relationships |
| **Class imbalance** | `smote`, `adasyn` | Designed for oversampling |
| **Privacy-sensitive** | `dp` | Formal privacy guarantees |
| **Time series / Sequential** | `par`, `timegan`, `dgan` | Temporal dependencies |
| **Single-cell / Gene expression** | `scvi`, `scgen` | Biological structure aware |
| **Fast prototyping** | `resample`, `cart` | Instant results |
| **Numeric-only data** | `gmm`, `diffusion` | Simple distributions |

### Single-Cell Methods Details

#### scVI (Single-cell Variational Inference)

Best for generating new single-cell-like observations from scratch. These methods are specifically designed for high-dimensional **transcriptomic data (RNA-seq)**. They use deep generative models to represent biological variation while handling the heavy sparsity and technical noise (dropout) typical of single-cell datasets. They are excellent for correcting "batch effects" and synthesizing coherent gene expression profiles.

**Input Format:** Accepts both `pd.DataFrame` and `AnnData` objects directly.

**DataFrame Input:**
```python
synthetic = gen.generate(
    data=expression_df,      # Rows=cells, Columns=genes
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Optional metadata column
    model_params={
        "epochs": 100,
        "n_latent": 10,      # Latent space dimensions
        "n_layers": 1,       # Encoder/decoder depth
    }
)
```

**AnnData Input (Recommended for single-cell data):**
```python
import anndata

# Create or load AnnData object
adata = anndata.read_h5ad("single_cell_data.h5ad")

synthetic = gen.generate(
    data=adata,              # Pass AnnData directly
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Must be in adata.obs
    model_params={
        "epochs": 100,
        "n_latent": 10,
        "n_layers": 1,
    }
)
# Returns pd.DataFrame with gene columns + metadata
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `n_latent` | 10 | Latent space dimensionality |
| `n_layers` | 1 | Number of hidden layers |

> [!NOTE]
> **Why scVI + scGen?** Together, these methods provide a comprehensive suite for single-cell synthesis. **scVI** is the gold standard for representing biological variance and generating "unbiased" cell populations, while **scGen** excels at predicting response to treatments and experimental conditions.

> **AnnData Support:** When passing `AnnData`, the object is used directly without conversion, preserving the original structure. The output is always a `pd.DataFrame` containing both the gene expression and the observations metadata.

#### scGen (Perturbation Prediction)

Best for generating cells under different conditions or removing batch effects.

```python
synthetic = gen.generate(
    data=expression_df,
    n_samples=1000,
    method="scgen",
    target_col="cell_type",
    model_params={
        "epochs": 100,
        "n_latent": 10,
        "condition_col": "treatment",  # Required: condition/batch column
        "noise_scale": 0.1,            # Diversity noise
    }
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `n_latent` | 10 | Latent space dimensionality |
| `condition_col` | None | Column with condition/batch labels (required) |
| `noise_scale` | 0.1 | Noise for sample diversity |

> **Note:** If `condition_col` is not provided, scGen automatically falls back to scVI.

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

---

## Common Usage Scenarios (Quick Guide)

### 1. Time Series (Sequences)
*   **Independent Sequences (Multi-Entity):** Use `method="par"` (Probabilistic AutoRegressive, requires SDV deep learning).
    ```python
    gen.generate(data, method="par", model_params={"sequence_key": "user_id"})
    ```
*   **Future Forecasting:** Not the primary use case. Use `StreamGenerator` for infinite flows or manual date injection.

### 2. Classification and Regression (Supervised)
If you have a `target` column (e.g., price, churn) and the relationship $X \rightarrow Y$ is critical:
*   Use `method="lgbm"` (LightGBM) or `method="rf"` (Random Forest).
*   Always specify `target_col="column_name"`.
    ```python
    # The generator automatically detects if it's Regression or Classification
    gen.generate(data, target_col="price", method="lgbm") 
    ```

### 3. Clustering (Unsupervised)
If there's no clear target and you want to preserve natural data clusters:
*   Use `method="gmm"` (Gaussian Mixture Models) or `method="tvae"` (Variational Autoencoder).
    ```python
    gen.generate(data, method="tvae")
    ```

### 4. Multi-Label (Multiple Labels)
If a cell contains multiple values (e.g., `["A", "B", "C"]`) or string format `"A,B,C"`:
*   **Limitation:** Standard models don't handle lists within cells well.
*   **Solution:** Transform the column to **One-Hot Encoding** (multiple binary columns `is_A`, `is_B`) before passing to the generator. Tree-based models (`lgbm`, `cart`) will learn correlations between labels (e.g., if `is_A=1` often implies `is_B=1`).

### 5. Block Data (Partitioned Data)
If your data is logically fragmented (e.g., by Stores, Countries, Patients) and you want independent models for each:
*   Use **`RealBlockGenerator`** instead of `RealGenerator`.
    ```python
    block_gen = RealBlockGenerator()
    block_gen.generate(data, block_column="StoreID", method="cart") 
    ```
    *This trains a different model for each StoreID.*

### 6. Handling Imbalanced Data
If your target column has minority classes that you want to amplify:
*   **Automatic Balancing:** Use `balance_target=True`. The generator applies internal oversampling (SMOTE/RandomOverSampler) so the model learns equally from all classes.
    ```python
    gen.generate(data, target_col="fraud", balance_target=True, method="cart")
    ```
*   **Custom Distribution:** If you want a specific ratio (e.g., 70% Class A, 30% Class B):
    ```python
    gen.generate(data, target_col="status", custom_distributions={"status": {"Low": 0.7, "High": 0.3}})
    ```
    *Note: `balance_target=True` is a shortcut for `custom_distributions={"col": "balanced"}`. For extreme imbalances, Deep Learning methods like `method="ctgan"` usually provide better stability than tree-based methods.*
---
