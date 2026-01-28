# Documentación de Calm Data Generator

Bienvenido a la documentación completa de **Calm Data Generator**. Esta guía cubre la instalación, configuración y uso avanzado de todos los módulos.

> **Nota:** Para documentos de referencia de API específicos, ver:
> - [RealGenerator API](./REAL_GENERATOR_REFERENCE_ES.md)
> - [DriftInjector API](./DRIFT_INJECTOR_REFERENCE_ES.md)
> - [StreamGenerator API](./STREAM_GENERATOR_REFERENCE_ES.md)
> - [ClinicalGenerator API](./CLINICAL_GENERATOR_REFERENCE_ES.md)
> - [Índice API](./API_ES.md)

---

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Generador Real (Tabular)](#realgenerator)
4. [Generador Clínico](#clinicalgenerator)
5. [Generador de Stream](#streamgenerator)
6. [Inyector de Drift](#driftinjector)
7. [Privacidad y Anonimización](#privacidad-y-anonimización)
8. [Generadores de Bloques](#generadores-de-bloques)
9. [Informes de Calidad](#informes-de-calidad)

---

## Instalación

```bash
# Instalación estándar
pip install calm-data-generator

# Con extras opcionales
pip install calm-data-generator[stream,timeseries,deeplearning]
```

Ver [README_ES.md](../../README_ES.md) para consejos de solución de problemas más detallados.

---

## Inicio Rápido

Ver [README_ES.md](../../README_ES.md) para ejemplos básicos de código.

---

## RealGenerator

**Clase:** `calm_data_generator.generators.tabular.RealGenerator`

El motor principal para generar datos sintéticos que imitan datasets tabulares reales.

### Uso Básico

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()
synthetic_data = gen.generate(real_data, n_samples=1000, method='lgbm')
```

### Métodos Soportados

| Método | Descripción | Caso de Uso |
|--------|-------------|-------------|
| `cart` | Árboles de Clasificación y Regresión | Iteración rápida, captura estructura básica. |
| `rf` | Random Forest | Mejor calidad que CART, más lento. |
| `lgbm` | LightGBM | Alta eficiencia y rendimiento para tablas grandes. |
| `ctgan` | Conditional GAN (SDV) | Deep learning para distribuciones complejas multi-modales. |
| `tvae` | Variational Autoencoder (SDV) | A menudo más rápido y robusto que GANs para datos tabulares. |
| `copula` | Cópula Gaussiana | Captura correlaciones estadísticas simples. Muy rápido. |
| `diffusion` | Difusión Tabular (DDPM) | Estado del arte experimental. Lento pero alta fidelidad. |
| `scgen/scvi` | Single-Cell (Genómica) | Modelado biológico especializado para RNA-Seq. |

### Configuración Avanzada (`model_params`)

Puedes pasar parámetros específicos al modelo subyacente a través de `model_params`.

**Para métodos basados en SDV (CTGAN, TVAE):**
- `epochs`: Número de épocas de entrenamiento (defecto: 300).
- `batch_size`: Tamaño del lote (defecto: 500).
- `cuda`: `True`/`False` para forzar uso de GPU.

**Para métodos basados en ML (LGBM, RF):**
- `n_estimators`: Número de árboles.
- `max_depth`: Profundidad máxima.
- `balance_target`: `True` para reequilibrar clases antes de entrenar.

---

## ClinicalGenerator

**Clase:** `calm_data_generator.generators.clinical.ClinicalDataGenerator`

Diseñado para simular datos sanitarios complejos incluyendo datos demográficos, genómicos (genes) y proteómicos (proteínas).

### Características Clave
- **Correlaciones Biológicas:** Simula dependencias realistas entre edad, género y expresión de biomarcadores.
- **Efectos de Enfermedad:** Permite inyectar señales específicas de enfermedad (ej. sobreexpresión de un gen).
- **Longitudinal:** Genera trayectorias de pacientes a lo largo del tiempo.

Ver [CLINICAL_GENERATOR_REFERENCE_ES.md](./CLINICAL_GENERATOR_REFERENCE_ES.md) para detalles completos de configuración.

---

## StreamGenerator

**Clase:** `calm_data_generator.generators.stream.StreamGenerator`

Un wrapper alrededor de la biblioteca `River` para generar flujos de datos infinitos con concept drift evolutivo.

### Flujo de Trabajo
1. Instanciar un generador de River (ej. `SEA`, `Agrawal`).
2. Pasarlo a `StreamGenerator.generate()`.
3. Aplicar drift, balanceo o inyección de fechas.

```python
from river import synth
from calm_data_generator.generators.stream import StreamGenerator

river_gen = synth.SEA()
gen = StreamGenerator()
df = gen.generate(river_gen, n_samples=5000)
```

Ver [STREAM_GENERATOR_REFERENCE_ES.md](./STREAM_GENERATOR_REFERENCE_ES.md).

---

## DriftInjector

**Clase:** `calm_data_generator.generators.drift.DriftInjector`

Permite modificar datasets existentes para introducir cambios estadísticos controlados (drift), útiles para probar sistemas de monitorización de ML.

### Tipos de Drift
- **Feature Drift:** Cambios en la distribución de las variables de entrada $P(X)$.
- **Label Drift:** Cambios en la distribución de la variable objetivo $P(y)$.
- **Concept Drift:** Cambios en la relación entre entrada y objetivo $P(y|X)$.

### Inyección Unificada

Usa `inject_drift()` para aplicar drift fácilmente a múltiples columnas sin preocuparte por sus tipos de datos.

```python
injector.inject_drift(df, columns=['salary'], drift_mode='gradual', drift_magnitude=0.5)
```

Ver [DRIFT_INJECTOR_REFERENCE_ES.md](./DRIFT_INJECTOR_REFERENCE_ES.md).

---

## Privacidad y Anonimización

Módulo: `calm_data_generator.privacy`

Herramientas para proteger información sensible antes de compartir datos.

### Funciones Principales

1.  **Pseudonimización**: Reemplaza identificadores con hashes o tokens reversibles.
    ```python
    df = pseudonymize_columns(df, columns=['user_id', 'email'])
    ```

2.  **Ruido Diferencial**: Añade ruido de Laplace para garantizar privacidad diferencial local.
    ```python
    df = add_laplace_noise(df, columns=['salary'], epsilon=1.0)
    ```

3.  **Generalización**: Agrupa valores precisos en rangos (ej. edad 23 -> "20-30").
    ```python
    df = generalize_numeric_to_ranges(df, col='age', min_val=0, max_val=100, step=10)
    ```

4.  **Shuffle**: Aleatoriza el orden de una columna para romper correlaciones con el resto de la fila (k-anonymity débil).

---

## Generadores de Bloques

Permiten crear datasets compuestos de múltiples partes ("bloques"), donde cada bloque puede representar un periodo de tiempo, ubicación o concepto diferente.

### Cómo Funciona

1.  **Partición**: Los datos de entrada se dividen en trozos basados en `block_column` (ej. Año, Región).
2.  **Modelado Independiente**: Se entrena un modelo generativo separado para **cada bloque**. Esto captura las propiedades estadísticas locales.
3.  **Generación**: Se generan datos sintéticos para cada bloque independientemente.
4.  **Ensamblaje**: Los bloques sintéticos se concatenan.

### Clases Soportadas

| Generador | Descripción |
|-----------|-------------|
| `RealBlockGenerator` | Divide un dataset real en bloques y aprende de cada uno. |
| `SyntheticBlockGenerator` | Concatena generadores de stream para simular drift sintético puro. |
| `ClinicalDataGeneratorBlock` | Genera datos clínicos multi-centro (ej. varios hospitales). |

### Ejemplo: RealBlockGenerator

```python
from calm_data_generator.generators.tabular import RealBlockGenerator

gen = RealBlockGenerator()

# Generar datos divididos por "Año"
synthetic_blocks = gen.generate(
    data=data,
    output_dir="./output",
    block_column="Year",
    target_col="Churn"
)
```

---

## Informes de Calidad

**Clase:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes HTML interactivos comparando los datos reales y sintéticos.

```python
from calm_data_generator.generators.tabular import QualityReporter

reporter = QualityReporter()
reporter.generate_report(real_df, synthetic_df, target_col='target')
```

**Métricas Incluidas:**
- **Estadísticas Descriptivas:** Comparación de media, std, min, max.
- **Distribuciones:** Histogramas superpuestos.
- **Correlaciones:** Mapas de calor de Pearson/Spearman.
- **PCA/TSNE:** Visualización de la variedad de datos en 2D.
- **Privacidad:** (Opcional) Tests de riesgo de reidentificación.
