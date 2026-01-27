# CALM-Data-Generator


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/calm-data-generator.svg)](https://badge.fury.io/py/calm-data-generator)

**CALM-Data-Generator** is a comprehensive Python library for synthetic data generation with advanced features for:
- **Clinical/Medical Data** - Generate realistic patient demographics, genes, proteins
- **Tabular Data Synthesis** - CTGAN, TVAE, Copula, CART, and more
- **Time Series** - TimeGAN, DGAN, PAR, Temporal Copula
- **Drift Injection** - Test ML model robustness with controlled drift
- **Privacy Preservation** - Differential privacy, pseudonymization, generalization
- **Scenario Evolution** - Feature evolution and target construction

## Core Technologies

This library leverages and unifies best-in-class open-source tools to provide a seamless data generation experience:

- **SDV (Synthetic Data Vault)**: The core engine for tabular deep learning models (CTGAN, TVAE) and statistical methods (Copula). **Included by default**.
  > **Note:** SDV versions 1.0+ use the Business Source License (BSL). While free for development and research, commercial production use may require a license from DataCebo. Please review their terms.
- **River**: Powers the streaming generation capabilities (`[stream]` extra).
- **Gretel Synthetics**: Provides advanced time-series generation via DoppelGANger (`[timeseries]` extra).
- **YData Profiling**: Generates comprehensive automated quality reports.
- **SmartNoise**: Enables differential privacy mechanisms.

## Safe Data Sharing

A key advantage of **Calm-Data-Generator** is enabling the use of private data in public or collaborative environments:

1.  **Private Origin**: You start with sensitive data (e.g., GDPR/HIPAA restricted) that cannot leave your secure environment.
2.  **Synthetic Twin**: The library generates a synthetic dataset that statistically mirrors the original but contains **no real individuals**.
3.  **Safe Distribution**: Once validated (using `QualityReporter`'s privacy checks), this synthetic dataset allows for **risk-free sharing**, model training, and testing without exposing confidential information.

## Key Use Cases

- **MLOps Monitoring Validation**: Use **StreamGenerator** and **DriftInjector** to simulate data drift (gradual, abrupt) and verify if your monitoring alerts trigger correctly before deployment.
- **Biomedical Research (HealthTech)**: Generate synthetic patient cohorts with **ClinicalDataGenerator** that preserve complex biological correlations (e.g., gene-age relationships) for collaborative studies without compromising patient privacy.
- **Stress Testing ("What-If" Analysis)**: Use **ScenarioInjector** to simulate future scenarios (e.g., "What if the customer age base increases by 10 years?") and measure model performance degradation under stress.
- **Development Data**: Provide developers with high-fidelity synthetic replicas of production databases, allowing them to build and test features safely without accessing sensitive real-world data.

---

## Architecture & Design

### Technical Architecture
Minimalist view of the system's core components and data flow.

![Architecture Diagram](calm_data_generator/docs/assets/architecture.png)

### Ecosystem Integration
How **Calm Data Generator** integrates with foundational libraries to provide enhanced value (Drift, Clinical Logic, Reporting).

![Ecosystem Diagram](calm_data_generator/docs/assets/ecosystem.png)

---

## Installation

```bash
# Basic installation
pip install calm-data-generator

# For Stream Generator (River)
pip install calm-data-generator[stream]

# For Time Series (Gretel Synthetics)
pip install calm-data-generator[timeseries]

# Full installation
pip install calm-data-generator[full]
```

**From source:**
```bash
git clone https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator.git
cd Calm-Data_Generator
pip install -e .
```

### Troubleshooting

If you encounter errors installing `river` (C compilation errors):

```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Then retry installation
pip install calm-data-generator
```

---

## Quick Start

### Generate Synthetic Data from Real Dataset

```python
from calm_data_generator.generators.tabular import RealGenerator
import pandas as pd

# Your real dataset
data = pd.read_csv("your_data.csv")

# Initialize generator
gen = RealGenerator()

# Generate 1000 synthetic samples using CTGAN
# model_params accepts any hyperparameter supported by the underlying model
synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    target_col='label',
    model_params={
        'epochs': 300,           # Training epochs
        'batch_size': 500,       # Batch size
        'discriminator_steps': 1 # CTGAN-specific parameter
    }
)

print(f"Generated {len(synthetic)} samples")
```

### Generate Clinical Data

```python
from calm_data_generator.generators.clinical import ClinicalDataGenerator, DateConfig

gen = ClinicalDataGenerator()

# Generate patient data with genes and proteins
result = gen.generate(
    n_samples=100,
    n_genes=500,
    n_proteins=200,
    date_config=DateConfig(start_date="2024-01-01")
)

demographics = result['demographics']
genes = result['genes']
proteins = result['proteins']
```

### Inject Drift for ML Testing

```python
from calm_data_generator.generators.drift import DriftInjector

injector = DriftInjector()

# Inject gradual drift to test model robustness
drifted_data = injector.inject_feature_drift_gradual(
    df=data,
    feature_cols=['feature1', 'feature2'],
    drift_magnitude=0.5,
    drift_type='shift',  # Options: gaussian_noise, shift, scale
    start_index=50,
    auto_report=False
)
```

### Stream Data Generation

```python
from calm_data_generator.generators.stream import StreamGenerator

stream_gen = StreamGenerator()

# Generate a data stream with Concept Drift
stream_data = stream_gen.generate(
    n_chunks=10,
    chunk_size=1000,
    concept_drift=True,  # Simulate concept drift over time
    n_features=10
)

print(f"Generated stream with {len(stream_data)} total samples")
```

### Quality Reporting

```python
from calm_data_generator.generators.tabular import QualityReporter

# Generate a quality report comparing real vs synthetic data
reporter = QualityReporter()

reporter.generate_report(
    real_data=data,
    synthetic_data=synthetic,
    output_dir="./quality_report",
    target_col="target"
)
# Report saved to ./quality_report/report.html
```

---

## Modules

| Module | Import | Description |
|--------|--------|-------------|
| **Tabular** | `generators.tabular` | RealGenerator, QualityReporter |
| **Clinical** | `generators.clinical` | ClinicalDataGenerator, ClinicalDataGeneratorBlock |
| **Stream** | `generators.stream` | StreamGenerator, SyntheticBlockGenerator |
| **Blocks** | `generators.tabular` | RealBlockGenerator |
| **Drift** | `generators.drift` | DriftInjector |
| **Dynamics** | `generators.dynamics` | ScenarioInjector |
| **Anonymizer** | `anonymizer` | Privacy transformations |
| **Reports** | `reports` | Visualizer |

---

## Synthesis Methods

| Method | Type | Description | Requirements / Notes |
|--------|------|-------------|----------------------|
| `cart` | ML | CART-based iterative synthesis (fast) | Base installation |
| `rf` | ML | Random Forest synthesis | Base installation |
| `lgbm` | ML | LightGBM-based synthesis | Base installation (Requires `lightgbm`) |
| `ctgan` | DL | Conditional GAN for tabular data | Requires `sdv` (heavy DL dep) |
| `tvae` | DL | Variational Autoencoder | Requires `sdv` (heavy DL dep) |
| `copula` | Statistical | Gaussian Copula | Base installation |
| `diffusion` | DL | Tabular Diffusion (DDPM) | **Experimental**. Requires `calm-data-generator[deeplearning]` |
| `smote` | Augmentation | SMOTE oversampling | Base installation |
| `adasyn` | Augmentation | ADASYN adaptive sampling | Base installation |
| `dp` | Privacy | Differential Privacy (PATE-CTGAN) | Requires `smartnoise-synth` |
| `timegan` | Time Series | TimeGAN for sequences | **Manual Install**. Requires `ydata-synthetic` & `tensorflow` (conflicts likely) |
| `dgan` | Time Series | DoppelGANger | Requires `calm-data-generator[timeseries]` (`gretel-synthetics`) |
| `par` | Time Series | Probabilistic AutoRegressive | Requires `sdv` |
| `copula_temporal` | Time Series | Gaussian Copula with temporal lags | Base installation |

---

## CLI Access

```bash
# List all tutorials
calm-data-generator tutorials

# Show a specific tutorial
calm-data-generator tutorials show 1

# Run a tutorial
calm-data-generator tutorials run 1

# Show version
calm-data-generator version
```

---

## Tutorials

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | Real Generator | Tabular data synthesis |
| 2 | Clinical Generator | Clinical/medical data |
| 3 | Drift Injector | Drift injection for ML |
| 4 | Stream Generator | Stream-based generation |
| 5 | Privacy | Privacy transformations |
| 6 | Scenario Injector | Feature evolution |

---

## License

MIT License - see [LICENSE](LICENSE) file

---

