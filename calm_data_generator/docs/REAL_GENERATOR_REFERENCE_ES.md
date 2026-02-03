# RealGenerator - Referencia Completa

**Ubicación:** `calm_data_generator.generators.tabular.RealGenerator`

El generador principal para la síntesis de datos tabulares a partir de datasets reales.

---

## Inicialización

```python
from calm_data_generator import RealGenerator

gen = RealGenerator(
    auto_report=True,       # Generar informe automáticamente tras síntesis
    minimal_report=False,   # Si es True, informe más rápido sin correlaciones/PCA
    random_state=42,        # Semilla para reproducibilidad
    logger=None,            # Logger de Python personalizado opcional
)
```

### Parámetros del Constructor

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `auto_report` | bool | `True` | Generar informe de calidad automáticamente |
| `minimal_report` | bool | `False` | Informe simplificado (más rápido) |
| `random_state` | int | `None` | Semilla para reproducibilidad |
| `logger` | Logger | `None` | Instancia de Logger de Python personalizada |

---

## Método Principal: `generate()`

```python
synthetic_df = gen.generate(
    data=df,                          # DataFrame original (requerido)
    n_samples=1000,                   # Número de muestras a generar (requerido)
    method="ctgan",                   # Método de síntesis
    target_col="target",              # Columna objetivo (opcional)
    output_dir="./output",            # Directorio de salida
    generator_name="my_generator",    # Nombre base para archivos de salida
    save_dataset=False,               # Guardar CSV resultante
    # Parámetros del modelo
    model_params={...},               # Parámetros específicos del método
    # Distribuciones personalizadas
    custom_distributions={"target": {0: 0.3, 1: 0.7}},
    # Inyección de fechas
    date_col="date",
    date_start="2024-01-01",
    date_step={"days": 1},
    # Post-procesamiento
    drift_injection_config=[...],
    dynamics_config={...},
    constraints=[...],
    adversarial_validation=True,      # Activar validación adversaria
)
```

### Parámetros de `generate()`

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `data` | DataFrame | - | Dataset original (requerido) |
| `n_samples` | int | - | Número de muestras a generar (requerido) |
| `method` | str | `"cart"` | Método de síntesis |
| `target_col` | str | `None` | Columna objetivo para balanceo |
| `output_dir` | str | `None` | Directorio para archivos de salida |
| `generator_name` | str | `"RealGenerator"` | Nombre base para archivos de salida |
| `save_dataset` | bool | `False` | Guardar dataset generado como CSV |
| `custom_distributions` | Dict | `None` | Distribución forzada por columna |
| `date_col` | str | `None` | Nombre de columna de fecha a inyectar |
| `date_start` | str | `None` | Fecha de inicio ("YYYY-MM-DD") |
| `date_step` | Dict | `None` | Incremento temporal (ej., `{"days": 1}`) |
| `date_every` | int | `1` | Incrementar fecha cada N filas |
| `drift_injection_config` | List[Dict] | `None` | Configuración de drift post-generación |
| `dynamics_config` | Dict | `None` | Configuración de evolución dinámica |
| `model_params` | Dict | `None` | Hiperparámetros específicos (pasa `**kwargs` al modelo) |
| `constraints` | List[Dict] | `None` | Restricciones de integridad |
| `adversarial_validation` | bool | `False` | Activar reporte de discriminador (Real vs Sintético) |

---

## Referencia Completa de `model_params`

El diccionario `model_params` permite el ajuste fino de parámetros internos para cada método de síntesis.

### Deep Learning (SDV)

| Parámetro | Métodos | Descripción |
|-----------|---------|-------------|
| `epochs` | Todos SDV | Número de épocas de entrenamiento |
| `batch_size` | Todos SDV | Tamaño del batch de entrenamiento |
| `verbose` | Todos SDV | Habilitar logs detallados |
| `**kwargs` | Todos | Cualquier parámetro soportado por el modelo subyacente (ej., `discriminator_steps` para CTGAN) |

**Ejemplo:**
```python
model_params={
    "epochs": 500, 
    "batch_size": 256,
    "discriminator_steps": 5  # Específico de CTGAN
}
```

### Machine Learning Clásico (CART, RF, LGBM)

| Parámetro | Métodos | Descripción |
|-----------|---------|-------------|
| `balance_target` | Todos ML | Si es True y `target_col` existe, balancea clases antes de entrenar |
| `n_estimators` | RF, LGBM | Número de árboles |
| `max_depth` | CART, RF | Profundidad máxima |

**Ejemplo:**
```python
method="rf",
target_col="churn",
model_params={
    "balance_target": True,
    "n_estimators": 100
}
```

### Single-Cell (scVI, scGen)

Estos métodos están diseñados específicamente para **datos transcriptómicos (RNA-seq)**. Utilizan modelos generativos profundos para manejar la dispersión (sparsity) y el ruido técnico característico de los datos biológicos. Son ideales para corregir "efectos de lote" (batch effects) y generar perfiles de expresión genética sintéticos coherentes.

**Formato de Entrada:** Acepta tanto `pd.DataFrame` como objetos `AnnData` directamente.

#### scVI (Single-cell Variational Inference)

**Entrada DataFrame:**
```python
synthetic = gen.generate(
    data=expression_df,      # Filas=células, Columnas=genes
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Columna de metadatos opcional
    model_params={
        "epochs": 100,
        "n_latent": 10,      # Dimensiones del espacio latente
        "n_layers": 1,       # Profundidad encoder/decoder
    }
)
```

**Formato de Entrada:** Acepta objetos `pd.DataFrame`, `AnnData` o rutas de archivo (`.h5` o `.h5ad`) directamente.

**Uso de Rutas de Archivo (H5/H5AD):**
```python
# ¡El generador carga el archivo automáticamente por ti!
synthetic = gen.generate(
    data="datos_single_cell.h5ad",  # O .h5
    n_samples=1000,
    method="scvi",
    target_col="cell_type"
)
```

**Entrada AnnData (Recomendado para datos single-cell):**
```python
import anndata

# Crear o cargar objeto AnnData
adata = anndata.read_h5ad("single_cell_data.h5ad")

synthetic = gen.generate(
    data=adata,              # Pasar AnnData directamente
    n_samples=1000,
    method="scvi",
    target_col="cell_type",  # Debe estar en adata.obs
    model_params={
        "epochs": 100,
        "n_latent": 10,
        "n_layers": 1,
    }
)
# Retorna pd.DataFrame con columnas de genes + metadatos
```

| Parámetro | Descripción |
|-----------|-------------|
| `epochs` | Épocas de entrenamiento (default: 100) |
| `n_latent` | Dimensionalidad del espacio latente (default: 10) |
| `n_layers` | Número de capas ocultas (default: 1) |

> [!NOTE]
> **¿Por qué scVI + scGen?** Juntos, estos métodos proporcionan una suite completa para la síntesis de datos single-cell. **scVI** es el estándar de oro para representar la varianza biológica y generar poblaciones celulares "imparciales", mientras que **scGen** destaca en la predicción de respuestas a tratamientos y condiciones experimentales.

> **Soporte AnnData:** Al pasar un objeto `AnnData`, este se utiliza directamente sin conversión, preservando la estructura original. El resultado es siempre un `pd.DataFrame` que contiene tanto la expresión génica como los metadatos de las observaciones (`obs`).

#### scGen (Predicción de Perturbaciones)

Mejor para generar células bajo diferentes condiciones o eliminar efectos de lote.

```python
synthetic = gen.generate(
    data=expression_df,
    n_samples=1000,
    method="scgen",
    target_col="cell_type",
    model_params={
        "epochs": 100,
        "n_latent": 10,
        "condition_col": "treatment",  # Requerido: columna de condición/lote
    }
)
```

| Parámetro | Descripción |
|-----------|-------------|
| `epochs` | Épocas de entrenamiento (default: 100) |
| `n_latent` | Dimensionalidad del espacio latente (default: 10) |
| `condition_col` | Columna con etiquetas de condición/lote (requerido) |

> **Nota:** Si no se proporciona `condition_col`, scGen automáticamente vuelve a scVI.

---

## Manejo de Datos Desbalanceados

`RealGenerator` ofrece varias estrategias para trabajar con datasets fuertemente desbalanceados (ej. detección de fraude, diagnósticos raros):

### 1. Re-balanceo Automático (`balance_target=True`)
Utiliza técnicas de re-muestreo antes o durante el entrenamiento para generar un dataset sintético equilibrado.
*   **Ideal para:** Entrenar modelos de clasificación robustos que requieren clases balanceadas.
*   **Comportamiento:** Si el original es 99% clase A y 1% clase B, el resultado será aprox. 50% A y 50% B.
*   **Métodos compatibles:** `cart`, `rf`, `lgbm`.

### 2. Control Manual de Distribución
Puede forzar la distribución de la clase objetivo usando `DriftInjector`.
*   **Ideal para:** Escenarios "What-If" (ej. "¿Qué pasa si el fraude aumenta al 10%?").
*   **Método:** `DriftInjector.inject_label_shift` post-generación.

### 3. Técnicas de Oversampling
Métodos clásicos para aumentar la clase minoritaria mediante interpolación.
*   **Métodos:** `smote` (Synthetic Minority Over-sampling Technique), `adasyn` (Adaptive Synthetic Sampling).
*   **Ideal para:** Datasets numéricos pequeños donde se necesita aumentar la representación de casos raros.

### 4. Fidelidad Estadística (Por defecto)
Si no se especifica ninguna opción, los modelos generativos avanzados (`ctgan`, `tvae`) aprenderán y replicarán la distribución original, preservando el desbalance real.
*   **Ideal para:** Análisis exploratorio fiel a la realidad o validación de sistemas en condiciones reales.

---

## Métodos Soportados

| Método | Tipo | Descripción |
|--------|------|-------------|
| `cart` | ML | Árboles de Clasificación y Regresión (Rápido, bueno para estructura) |
| `rf` | ML | Random Forest (Robusto, más lento que CART) |
| `lgbm` | ML | LightGBM (Gradient Boosting, muy eficiente) |
| `ctgan` | DL | Conditional GAN para tablas (Estándar de la industria) |
| `tvae` | DL | Variational Autoencoder para tablas (Más rápido que GANs) |
| `copula` | Est. | Cópula Gaussiana (Captura correlaciones estadísticas simples) |
| `smote` | Aug. | Synthetic Minority Over-sampling Technique |
| `adasyn` | Aug. | Adaptive Synthetic Sampling |
| `scvi` | Gen. | Variational Inference para datos single-cell |
| `scgen` | Gen. | Predicción de perturbaciones para datos single-cell |
| `dp` | Priv. | Privacidad Diferencial (Requiere SmartNoise) |

---

## Escenarios de Uso Comunes (Guía Rápida)

### 1. Series Temporales (Time Series)
*   **Secuencias Independientes (Multi-Entity):** Usa `method="par"` (Probabilistic AutoRegressive, requiere SDV con deep learning).
    ```python
    gen.generate(data, method="par", model_params={"sequence_key": "user_id"})
    ```
*   **Proyección de Futuro (Forecasting):** No es el caso de uso principal. Usa `StreamGenerator` para flujos infinitos o inyección de fechas manual.

### 2. Clasificación y Regresión (Supervisado)
Si tienes una columna `target` (ej. precio, churn) y la relación $X \rightarrow Y$ es crítica:
*   Usa `method="lgbm"` (LightGBM) o `method="rf"` (Random Forest).
*   Especifica siempre `target_col="nombre_columna"`.
    ```python
    # El generador detecta automáticamente si es Regresión o Clasificación
    gen.generate(data, target_col="precio", method="lgbm") 
    ```

### 3. Clustering (No Supervisado)
Si no hay un target claro y quieres preservar grupos naturales de datos:
*   Usa `method="gmm"` (Gaussian Mixture Models, vía librería externa si disponible) o `method="tvae"` (Variational Autoencoder).
    ```python
    gen.generate(data, method="tvae")
    ```

### 4. Multi-Label (Etiquetas Múltiples)
Si una celda contiene múltiples valores (ej: `["A", "B", "C"]`) o formato string `"A,B,C"`:
*   **Limitación:** Los modelos estándar no manejan bien listas dentro de celdas.
*   **Solución:** Transforma la columna a **One-Hot Encoding** (múltiples columnas binarias `is_A`, `is_B`) antes de pasarla al generador. Los modelos basados en árboles (`lgbm`, `cart`) aprenderán las correlaciones entre etiquetas (ej: si `is_A=1` suele implicar `is_B=1`).

### 5. Datos por Bloques (Blocks)
Si tus datos están fragmentados lógicamente (ej: por Tiendas, Países, Pacientes) y quieres modelos independientes para cada uno:
*   Usa **`RealBlockGenerator`** en lugar de `RealGenerator`.
    ```python
    block_gen = RealBlockGenerator()
    block_gen.generate(data, block_column="TiendaID", method="cart") 
    ```
    *Esto entrena un modelo diferente para cada TiendaID.*

### 6. Manejo de Datos Desbalanceados (Imbalance)
Si tu columna objetivo (`target`) tiene clases muy minoritarias que quieres potenciar:
*   **Balanceo Automático:** Usa `balance_target=True`. El generador aplicará técnicas de sobremuestreo (SMOTE/RandomOverSampler) internamente para que el modelo aprenda por igual de todas las clases.
    ```python
    gen.generate(data, target_col="fraude", balance_target=True, method="cart")
    ```
*   **Distribución Personalizada:** Si quieres una proporción exacta (ej: 70% Clase A, 30% Clase B):
    ```python
    gen.generate(data, target_col="nivel", custom_distributions={"nivel": {"Bajo": 0.7, "Alto": 0.3}})
    ```
    *Nota: `balance_target` es un atajo para `custom_distributions={"col": "balanced"}`. Para desbalanceos extremos, los métodos de Deep Learning como `method="ctgan"` suelen ofrecer mayor estabilidad que los métodos basados en árboles.*
---
