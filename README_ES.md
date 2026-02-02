# CALM-Data-Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/calm-data-generator.svg)](https://badge.fury.io/py/calm-data-generator)

> 🇬🇧 **[English README](README.md)**

**CALM-Data-Generator** es una biblioteca completa en Python para la generación de datos sintéticos con características avanzadas para:
- **Datos Clínicos/Médicos** - Genera demografía de pacientes, genes y proteínas realistas.
- **Síntesis Tabular** - CTGAN, TVAE, Copula, CART y más.
- **Series Temporales** - TimeGAN, DGAN, PAR, Cópula Temporal.
- **Inyección de Drift (Desviación)** - Prueba la robustez de modelos ML con drift controlado.
- **Preservación de Privacidad** - Privacidad diferencial, pseudonimización, generalización.
- **Evolución de Escenarios** - Evolución de features y construcción de targets.

## Alcance y Capacidades

**Calm-Data-Generator** está optimizado para **datos tabulares estructurados**. Está diseñado para manejar:
- ✅ **Clasificación** (Binaria y Multiclase)
- ✅ **Regresión** (Variables continuas)
- ✅ **Multi-label** (Múltiples objetivos)
- ✅ **Clustering** (Preservación de agrupamientos naturales)
- ✅ **Series Temporales** (Correlaciones y patrones temporales)
- ✅ **Single-Cell / Genómica** (Datos de expresión RNA-seq)

> [!IMPORTANT]
> Esta biblioteca **NO** está diseñada para datos no estructurados como **Imágenes**, **Vídeos** o **Audio**. No incluye modelos de Visión Artificial o Procesamiento de Señales.

---

## Tecnologías Principales

Esta biblioteca aprovecha y unifica las mejores herramientas de código abierto para proporcionar una experiencia de generación de datos fluida:

- **SDV (Synthetic Data Vault)**: El motor principal para modelos tabulares de deep learning (CTGAN, TVAE) y métodos estadísticos (Copula). **Incluido por defecto**.
  > **Nota:** Las versiones de SDV 1.0+ usan la licencia Business Source License (BSL). Aunque es libre para desarrollo e investigación, el uso comercial en producción puede requerir una licencia de DataCebo. Por favor revisa sus términos.
- **River**: Potencia las capacidades de generación en streaming (`[stream]` extra).
- **Gretel Synthetics**: Proporciona generación avanzada de series temporales vía DoppelGANger (`[timeseries]` extra).
- **YData Profiling**: Genera informes de calidad automatizados y completos.
- **SmartNoise**: Habilita mecanismos de privacidad diferencial.

## Intercambio Seguro de Datos

Una ventaja clave de **Calm-Data-Generator** es permitir el uso de datos privados en entornos públicos o colaborativos:

1.  **Origen Privado**: Empiezas con datos sensibles (ej. restringidos por GDPR/HIPAA) que no pueden salir de tu entorno seguro.
2.  **Gemelo Sintético**: La biblioteca genera un conjunto de datos sintético que refleja estadísticamente el original pero **no contiene individuos reales**.
3.  **Distribución Segura**: Una vez validado (usando los chequeos de privacidad de `QualityReporter`), este dataset sintético permite **compartir sin riesgos**, entrenar modelos y realizar pruebas sin exponer información confidencial.

## Casos de Uso Clave

- **Validación de Monitorización MLOps**: Usa **StreamGenerator** y **DriftInjector** para simular drift de datos (gradual, abrupto) y verificar si tus alertas de monitorización se activan correctamente antes del despliegue.
- **Investigación Biomédica (HealthTech)**: Genera cohortes de pacientes sintéticos con **ClinicalDataGenerator** que preservan correlaciones biológicas complejas (ej. relaciones gen-edad) para estudios colaborativos sin comprometer la privacidad del paciente.
- **Pruebas de Estrés (Análisis "What-If")**: Usa **ScenarioInjector** para simular escenarios futuros (ej. "¿Qué pasa si la base de clientes envejece 10 años?") y medir la degradación del rendimiento del modelo bajo estrés.
- **Datos de Desarrollo**: Proporciona a los desarrolladores réplicas sintéticas de alta fidelidad de bases de datos de producción, permitiéndoles construir y probar funcionalidades de forma segura sin acceder a datos reales sensibles.

---

## Instalación

```bash
# Instalación básica
pip install calm-data-generator

# Para Stream Generator (River)
pip install calm-data-generator[stream]

# Para Series Temporales (Gretel Synthetics)
pip install calm-data-generator[timeseries]

# Instalación completa
pip install calm-data-generator[full]
```

**Desde el código fuente:**
```bash
git clone https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator.git
cd Calm-Data_Generator
pip install .
```

### Solución de Problemas

**Zsh shell (macOS/Linux):** Si los corchetes dan error, usa comillas:
```bash
pip install "calm-data-generator[stream]"
```

**Errores de compilación de River (Linux/macOS):**
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Luego reintenta
pip install calm-data-generator
```

**Usuarios de Windows:** Instala Visual Studio Build Tools primero:
1. Descarga [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Instala "Desktop development with C++"
3. Luego reintenta la instalación

**PyTorch solo-CPU (sin GPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install calm-data-generator
```

---

## Inicio Rápido

### Generar Datos Sintéticos desde un Dataset Real

```python
from calm_data_generator import RealGenerator
import pandas as pd

# Tu dataset real
data = pd.read_csv("your_data.csv")

# Inicializar generador
gen = RealGenerator()

# Generar 1000 muestras sintéticas usando CTGAN
# model_params acepta cualquier hiperparámetro soportado por el modelo subyacente
synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    target_col='label',
    model_params={
        'epochs': 300,           # Épocas de entrenamiento
        'batch_size': 500,       # Tamaño del batch
        'discriminator_steps': 1 # Parámetro específico de CTGAN
    }
)

print(f"Generadas {len(synthetic)} muestras")
```

### Aceleración por GPU

**Métodos con soporte GPU:**

| Método | Soporte GPU | Parámetro |
|--------|-------------|-----------|
| `ctgan`, `tvae`, `copula` | ✅ CUDA/MPS | `enable_gpu=True` |
| `par` (series temporales) | ✅ CUDA/MPS | `enable_gpu=True` |
| `dgan` (DoppelGANger) | ✅ PyTorch | Auto-detectado |
| `diffusion` | ✅ PyTorch | Auto-detectado |
| `smote`, `adasyn`, `cart`, `rf`, `lgbm`, `gmm`, `dp`, `datasynth` | ❌ Solo CPU | - |

```python
synthetic = gen.generate(
    data=data,
    n_samples=1000,
    method='ctgan',
    model_params={
        'epochs': 300,
        'enable_gpu': True  # GPU explícita - auto-detectado por defecto
    }
)
```

### Generar Datos Clínicos

```python
from calm_data_generator import ClinicalDataGenerator
from calm_data_generator.generators.configs import DateConfig

gen = ClinicalDataGenerator()

# Generar datos de pacientes con genes y proteínas
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

### Inyectar Drift para Pruebas de ML

**Opción 1: Directamente desde `generate()` (recomendado)**

```python
from calm_data_generator import RealGenerator

gen = RealGenerator()

# Generar datos sintéticos CON drift en una sola llamada
synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    method='ctgan',
    target_col='label',
    drift_injection_config=[
        {
            "method": "inject_drift",
            "params": {
                "columns": ["age", "income", "label"],
                "drift_mode": "gradual", # Auto-detecta tipos de columna
                "drift_magnitude": 0.3,
                "center": 500,
                "width": 200
            }
        }
    ]
)
```

**Opción 2: DriftInjector Independiente**

```python
from calm_data_generator import DriftInjector

injector = DriftInjector()

# Inyección unificada de drift (auto-detecta tipos)
drifted_data = injector.inject_drift(
    df=data,
    columns=['feature1', 'feature2', 'status'],
    drift_mode='gradual',
    drift_magnitude=0.5,
    # Configuración específica opcional
    numeric_operation='shift',
    categorical_operation='frequency',
    boolean_operation='flip'
)
```

**Métodos de drift disponibles:** `inject_drift` (unificado), `inject_feature_drift_gradual`, `inject_label_drift`, `inject_categorical_frequency_drift`, y más. Ver [DRIFT_INJECTOR_REFERENCE.md](calm_data_generator/docs/DRIFT_INJECTOR_REFERENCE.md).

### Simulación de Streaming

```python
from calm_data_generator import StreamGenerator

# Simular un stream de datos basándose en el dataset real
stream_gen = StreamGenerator()

stream_data = stream_gen.generate(
    data=data,
    n_samples=5000,
    chunk_size=1000,
    concept_drift=True,  # Simular concept drift en el tiempo
    n_features=10
)

print(f"Generado stream con {len(stream_data)} muestras totales")
```

### Informe de Calidad

```python
from calm_data_generator import QualityReporter

# Generar informe comparando datos reales vs sintéticos
reporter = QualityReporter()

reporter.generate_report(
    real_data=data,
    synthetic_data=synthetic,
    output_dir="./quality_report",
    target_col="target"
)
# Informe guardado en ./quality_report/report.html
```

---

## Módulos

| Módulo | Importación | Descripción |
|--------|-------------|-------------|
| **Tabular** | `generators.tabular` | RealGenerator, QualityReporter |
| **Clinical** | `generators.clinical` | ClinicalDataGenerator, ClinicalDataGeneratorBlock |
| **Stream** | \`generators.stream\` | StreamGenerator, StreamBlockGenerator |
| **Blocks** | `generators.tabular` | RealBlockGenerator |
| **Drift** | `generators.drift` | DriftInjector |
| **Dynamics** | `generators.dynamics` | ScenarioInjector |
| **Anonymizer** | `anonymizer` | Transformaciones de privacidad |
| **Reports** | `reports` | Visualizer |

---

## Métodos de Síntesis

| Método | Tipo | Descripción | Requisitos / Notas |
|--------|------|-------------|--------------------|
| `cart` | ML | Síntesis iterativa basada en CART (rápido) | Instalación base |
| `rf` | ML | Síntesis con Random Forest | Instalación base |
| `lgbm` | ML | Síntesis basada en LightGBM | Instalación base (Requiere `lightgbm`) |
| `ctgan` | DL | Conditional GAN para tabular | Requiere `sdv` (dependencia DL pesada) |
| `tvae` | DL | Variational Autoencoder | Requiere `sdv` (dependencia DL pesada) |
| `copula` | Estadístico | Cópula Gaussiana | Instalación base |
| `diffusion` | DL | Difusión Tabular (DDPM) | **Experimental**. Requiere `calm-data-generator[deeplearning]` |
| `smote` | Aumento | Sobremuestreo SMOTE | Instalación base |
| `adasyn` | Aumento | Muestreo adaptativo ADASYN | Instalación base |
| `dp` | Privacidad | Privacidad Diferencial (PATE-CTGAN) | Requiere `smartnoise-synth` |
| `timegan` | Series Temp. | TimeGAN para secuencias | **Instalación Manual**. Requiere `ydata-synthetic` & `tensorflow` |
| `dgan` | Series Temp. | DoppelGANger | Requiere `calm-data-generator[timeseries]` (`gretel-synthetics`) |
| `par` | Series Temp. | Probabilistic AutoRegressive | Requiere `sdv` |
| `copula_temporal` | Series Temp. | Cópula Gaussiana con lags temporales | Instalación base |
| `gmm` | Estadístico | Modelos de Mezcla Gaussiana | Instalación base |
| `datasynth` | Estadístico | DataSynthesizer (Greedy Bayes) | Requiere `DataSynthesizer` |
| `scvi` | Single-Cell | scVI (Variational Inference) para RNA-seq | Requiere `scvi-tools` |
| `scgen` | Single-Cell | scGen (Predictor de Perturbaciones) | Requiere `scvi-tools` |

---

## Documentación e Índice

Explora la documentación completa en el directorio `calm_data_generator/docs/`:

| Documento | Descripción |
|-----------|-------------|
| **[DOCUMENTATION.md](calm_data_generator/docs/DOCUMENTATION.md)** | **Guía Principal**. Manual completo cubriendo todos los módulos, conceptos y uso avanzado. |
| **[REAL_GENERATOR_REFERENCE.md](calm_data_generator/docs/REAL_GENERATOR_REFERENCE.md)** | **Referencia API para `RealGenerator`**. Parámetros detallados para todos los métodos de síntesis (`ctgan`, `lgbm`, `scvi`, etc.). |
| **[DRIFT_INJECTOR_REFERENCE.md](calm_data_generator/docs/DRIFT_INJECTOR_REFERENCE.md)** | **Referencia API para `DriftInjector`**. Guía para usar `inject_drift` y capacidades especializadas de drift. |
| **[STREAM_GENERATOR_REFERENCE.md](calm_data_generator/docs/STREAM_GENERATOR_REFERENCE.md)** | **Referencia API para `StreamGenerator`**. Detalles sobre simulación de stream e integración de drift. |
| **[CLINICAL_GENERATOR_REFERENCE.md](calm_data_generator/docs/CLINICAL_GENERATOR_REFERENCE.md)** | **Referencia API para `ClinicalGenerator`**. Configuración para genes, proteínas y datos de pacientes. |
| **[API.md](calm_data_generator/docs/API.md)** | **Índice Técnico de API**. Índice de alto nivel de clases y funciones. |

---

## Licencia

Licencia MIT - ver archivo [LICENSE](LICENSE)
