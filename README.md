# CALM-Data-Generator 🎲

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/calm-data-generator.svg)](https://badge.fury.io/py/calm-data-generator)

**CALM-Data-Generator** is a comprehensive Python library for synthetic data generation with advanced features for:
- 🏥 **Clinical/Medical Data** - Generate realistic patient demographics, genes, proteins
- 📊 **Tabular Data Synthesis** - CTGAN, TVAE, Copula, CART, and more
- 🌊 **Time Series** - TimeGAN, DGAN, PAR, Temporal Copula
- 🔀 **Drift Injection** - Test ML model robustness with controlled drift
- 🔒 **Privacy Preservation** - Differential privacy, pseudonymization, generalization
- 🎯 **Scenario Evolution** - Feature evolution and target construction

---

## 📦 Installation

```bash
# Basic installation
pip install calm-data-generator

# With deep learning models (CTGAN, TVAE, TimeGAN)
pip install calm-data-generator[deeplearning]

# Full installation with all features
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

## 🚀 Quick Start

### Generate Synthetic Data from Real Dataset

```python
from calm_data_generator.generators.tabular import RealGenerator
import pandas as pd

# Your real dataset
data = pd.read_csv("your_data.csv")

# Initialize generator
gen = RealGenerator()

# Generate 1000 synthetic samples using CTGAN
synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    target_col='label'
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

---

## 📚 Modules

| Module | Import | Description |
|--------|--------|-------------|
| **Tabular** | `generators.tabular` | RealGenerator, QualityReporter |
| **Clinical** | `generators.clinical` | ClinicalDataGenerator |
| **Stream** | `generators.stream` | StreamGenerator |
| **Drift** | `generators.drift` | DriftInjector |
| **Dynamics** | `generators.dynamics` | ScenarioInjector |
| **Anonymizer** | `anonymizer` | Privacy transformations |
| **Reports** | `reports` | Visualizer |

---

## 🔬 Synthesis Methods

| Method | Type | Description |
|--------|------|-------------|
| `cart` | ML | CART-based iterative synthesis (fast) |
| `rf` | ML | Random Forest synthesis |
| `lgbm` | ML | LightGBM-based synthesis |
| `ctgan` | DL | Conditional GAN for tabular data |
| `tvae` | DL | Variational Autoencoder |
| `copula` | Statistical | Gaussian Copula |
| `diffusion` | DL | Tabular Diffusion (DDPM) |
| `smote` | Augmentation | SMOTE oversampling |
| `adasyn` | Augmentation | ADASYN adaptive sampling |
| `dp` | Privacy | Differential Privacy (PATE-CTGAN) |
| `timegan` | Time Series | TimeGAN for sequences |
| `dgan` | Time Series | DoppelGANger |
| `par` | Time Series | Probabilistic AutoRegressive |
| `copula_temporal` | Time Series | Gaussian Copula with temporal lags |

---

## 🖥️ CLI Access

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

## 📖 Tutorials

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | Real Generator | Tabular data synthesis |
| 2 | Clinical Generator | Clinical/medical data |
| 3 | Drift Injector | Drift injection for ML |
| 4 | Stream Generator | Stream-based generation |
| 5 | Privacy | Privacy transformations |
| 6 | Scenario Injector | Feature evolution |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 📞 Contact

**Alejandro Belda Fernandez** - alejandrobeldafernandez@gmail.com

Project: [https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator](https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator)
