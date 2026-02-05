# DriftInjector - Referencia Completa

**Ubicación:** `calm_data_generator.generators.drift.DriftInjector`

Un módulo para inyectar varios tipos de drift (desplazamiento de datos) en datasets.

---

## Inicio Rápido: Drift desde `generate()`

Puedes inyectar drift directamente al generar datos sintéticos usando `RealGenerator.generate()`:

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()

synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    method='ctgan',
    drift_injection_config=[
        {
            "method": "inject_drift",
            "params": {
                "columns": ["age", "income"],
                "drift_mode": "gradual",
                "drift_magnitude": 0.3
            }
        }
    ]
)
```

Cada elemento en `drift_injection_config` requiere:
- `method`: Nombre del método de DriftInjector (ver abajo)
- `params`: Diccionario de parámetros para ese método

---

## Inicialización

```python
from calm_data_generator.generators.drift import DriftInjector

injector = DriftInjector(
    output_dir="./drift_output",      # Directorio de salida
    generator_name="my_drift",        # Nombre base para archivos
    random_state=42,                  # Semilla
    time_col="timestamp",             # Columna de tiempo por defecto
    block_column="block",             # Columna de bloque por defecto
    target_column="target",           # Columna objetivo por defecto
    auto_report=True,                 # Generar informe automáticamente
    minimal_report=False,             # Informes simplificados
)
```

### Parámetros del Constructor

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `output_dir` | str | `"drift_output"` | Directorio para salidas |
| `generator_name` | str | `"DriftInjector"` | Nombre base para archivos |
| `random_state` | int | `None` | Semilla para reproducibilidad |
| `time_col` | str | `None` | Columna de tiempo por defecto |
| `block_column` | str | `None` | Columna de bloque por defecto |
| `target_column` | str | `None` | Columna objetivo por defecto |
| `auto_report` | bool | `True` | Generar informe de calidad auto. |
| `minimal_report` | bool | `False` | Informes simplificados |

---

## Inyección de Drift Unificada: `inject_drift()`

**¡NUEVO!** Un único método que auto-detecta tipos de columna y aplica operaciones de drift apropiadas.

```python
drifted = injector.inject_drift(
    df=data,
    columns=['age', 'income', 'gender', 'is_active'],  # Cualquier tipo
    drift_mode='gradual',          # 'abrupt', 'gradual', 'incremental', 'recurrent'
    drift_magnitude=0.3,
    center=500,                    # Para modo gradual
    width=200,
    correlations=True,             # Propagar drift a columnas correlacionadas
)
```

### Propagación de Correlaciones

La inyección de drift puede respetar la estructura de correlación de tus datos, asegurando que los cambios en una característica se reflejen de manera realista en las características correlacionadas.

**Parámetro `correlations`:**
- **`True`**: Calcula la matriz de correlación de Pearson desde el DataFrame actual y propaga el drift proporcionalmente.
- **`pd.DataFrame`** o **`Dict`**: Usa una matriz o diccionario de correlación específico proporcionado por ti.
- **`None`** (Defecto): No se realiza propagación; solo cambian las columnas especificadas.

Mecanismo: $\Delta Y = \rho_{XY} \cdot \frac{\sigma_Y}{\sigma_X} \cdot \Delta X$

### Tipos de Columna Auto-Detectados

| Tipo Columna | Detección | Operación por Defecto |
|--------------|-----------|-----------------------|
| **Numérica** | dtypes `int`, `float` | `shift` |
| **Categórica** | dtypes `object`, `category` | `frequency` |
| **Booleana** | dtype `bool` o 2 valores únicos | `flip` |

### Modos de Drift

| Modo | Descripción |
|------|-------------|
| `abrupt` | Cambio inmediato desde `start_index` |
| `gradual` | Transición suave usando función ventana (sigmoide, lineal, coseno) |
| `incremental` | Drift constante y suave sobre todo el rango |
| `recurrent` | Múltiples ventanas de drift (controlado por `repeats`) |

### Parámetros

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `columns` | List[str] | - | Columnas donde aplicar drift (cualquier tipo) |
| `drift_magnitude` | float | `0.3` | Intensidad del drift (0.0 a 1.0) |
| `drift_mode` | str | `"abrupt"` | Patrón del drift |
| `numeric_operation` | str | `"shift"` | Operación para columnas numéricas |
| `categorical_operation` | str | `"frequency"` | Operación para columnas categóricas |
| `boolean_operation` | str | `"flip"` | Operación para columnas booleanas |
| `center` | int | auto | Centro de ventana de transición (gradual) |
| `width` | int | auto | Ancho de ventana de transición (gradual) |
| `profile` | str | `"sigmoid"` | Perfil de transición: `sigmoid`, `linear`, `cosine` |
| `repeats` | int | `3` | Número de ventanas (recurrent) |
| `start_index` | int | `None` | Fila donde empieza el drift |
| `conditions` | List[Dict] | `None` | Filtros para drift condicional |

### Operaciones Disponibles

**Numéricas**: `shift` (desplazar), `scale` (escalar), `gaussian_noise` (ruido gaussiano), `uniform_noise` (ruido uniforme), `add_value` (sumar), `subtract_value`, `multiply_value`

**Categóricas**: 
- `frequency`: Cambia la distribución de frecuencia (hace lo raro frecuente).
- `new_category`: Introduce un valor nuevo (ej. "NEW_CAT").
- `typos`: Introduce errores tipográficos o ruido de caracteres.

**Booleanas**: 
- `flip`: Invierte el valor (True <-> False).

---

## Tipos de Operación (`drift_type`)

Para columnas **numéricas**:

| Tipo | Descripción | Fórmula |
|------|-------------|---------|
| `gaussian_noise` | Ruido Gaussiano | `x + N(0, magnitude * std)` |
| `uniform_noise` | Ruido Uniforme | `x + U(-mag*std, +mag*std)` |
| `shift` | Desplazamiento de Media | `x + (mean * magnitude)` |
| `scale` | Escalado (apertura) | `mean + (x - mean) * (1 + magnitude)` |
| `add_value` | Sumar valor fijo | `x + drift_value` |

---

## Métodos Especializados

Aunque `inject_drift` es recomendado, puedes usar métodos específicos para control granular.

**Numéricos:**
- `inject_feature_drift`
- `inject_feature_drift_gradual`
- `inject_feature_drift_incremental`
- `inject_feature_drift_recurrent`

**Targets/Labels:**
- `inject_label_drift` (Abrupto)
- `inject_label_drift_gradual`

**Categóricos:**
- `inject_categorical_frequency_drift`
- `inject_new_category_drift`
- `inject_typos_drift`

**Otros:**
- `inject_missing_values_drift` (Introduce NaNs)
- `inject_conditional_drift` (Drift basado en filtros SQL-like)

## Ejemplo: Drift Numérico No-Negativo

Para inyectar drift en columnas numéricas que deben permanecer positivas (ej. precios, edad), usa la operación `scale`. Esto multiplica los valores en lugar de sumar, preservando el signo (si el factor es positivo).

```python
# Evitar valores negativos (ej. Salario, Edad)
# Usa 'scale' (multiplicación) en lugar de 'shift' (suma).
drifted = injector.inject_drift(
    df=data,
    columns=['salary', 'age'],
    drift_mode='gradual',
    drift_magnitude=0.2,       # Aumenta valores un ~20%
    numeric_operation='scale'  # Seguro para datos no negativos
)
```
