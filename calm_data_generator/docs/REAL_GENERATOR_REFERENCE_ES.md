# RealGenerator - Referencia Completa

**Ubicación:** `calm_data_generator.generators.tabular.RealGenerator`

El generador principal para la síntesis de datos tabulares a partir de datasets reales.

---

## Inicialización

```python
from calm_data_generator.generators.tabular import RealGenerator

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

| Parámetro | Descripción |
|-----------|-------------|
| `n_hidden` | Número de neuronas ocultas |
| `n_layers` | Número de capas ocultas |
| `n_epochs` | Épocas de entrenamiento |
| `condition_col` | (Solo scGen/scVI) Columna que define condiciones experimentales o lotes |

**Ejemplo:**
```python
method="scvi",
model_params={
    "n_hidden": 128,
    "n_layers": 2,
    "condition_col": "batch"
}
```

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
