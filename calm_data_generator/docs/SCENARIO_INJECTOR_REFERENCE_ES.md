# Referencia del ScenarioInjector

**Ubicación:** `calm_data_generator.generators.dynamics.ScenarioInjector`

Un módulo para evolucionar variables (features), construir variables objetivo (targets) basadas en reglas y proyectar datos a periodos de tiempo futuros.

---

## Inicialización

```python
from calm_data_generator.generators.dynamics import ScenarioInjector

scenario = ScenarioInjector(
    seed=42,                    # Semilla para reproducibilidad
    minimal_report=False,       # Informes completos
)
```

### Parámetros del Constructor

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `seed` | int | `None` | Semilla para reproducibilidad |
| `minimal_report` | bool | `False` | Si es True, informes simplificados |

---

## Método: `evolve_features()`

Evoluciona columnas numéricas basándose en configuraciones como tendencias, estacionalidad, ruido, etc.

### Sintaxis Básica

```python
evolved_df = scenario.evolve_features(
    df=df,                                    # DataFrame de entrada
    evolution_config={...},                   # Config de evolución por columna
    time_col="date",                          # Columna de tiempo (opcional)
    output_dir="./output",                    # Directorio de salida
    auto_report=True,                         # Generar informe
    generator_name="ScenarioInjector",        # Nombre base del archivo
    resample_rule=None,                       # Regla de re-muestreo temporal
)
```

### Parámetros

| Parámetro | Tipo | Defecto | Descripción |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Dataset original (requerido) |
| `evolution_config` | Dict | - | Configuración de evolución por columna |
| `time_col` | str | `None` | Columna de tiempo para ordenar |
| `output_dir` | str | `None` | Directorio de salida |
| `auto_report` | bool | `True` | Generar informe automáticamente |
| `generator_name` | str | `"ScenarioInjector"` | Nombre base para archivos de salida |
| `resample_rule` | str/int | `None` | Regla de re-muestreo (ej., "D", "W") |

### Tipos de Evolución

#### 1. Tendencia Lineal (`trend`)

Añade una pendiente lineal creciente o decreciente:

```python
evolution_config = {
    "price": {
        "type": "trend",
        "slope": 0.01,      # Incremento por fila (positivo = crecimiento)
    }
}
```

**Fórmula:** `nuevo_valor = valor_antiguo + (pendiente * indice_fila)`

#### 2. Estacionalidad (`seasonal`)

Añade un patrón sinusoidal periódico:

```python
evolution_config = {
    "temperature": {
        "type": "seasonal",
        "amplitude": 5.0,     # Altura máxima desde el centro
        "period": 365,        # Filas por ciclo completo
        "phase": 0,           # Desfase opcional
    }
}
```

**Fórmula:** `nuevo_valor = valor_antiguo + amplitud * sin(2π * indice_fila / periodo + fase)`

#### 3. Ruido Gaussiano (`noise`)

Añade ruido aditivo aleatorio:

```python
evolution_config = {
    "reading": {
        "type": "noise",
        "scale": 0.1,         # Desviación estándar del ruido
    }
}
```

#### 4. Decaimiento Exponencial (`decay`)

Aplica un factor de decaimiento exponencial:

```python
evolution_config = {
    "battery": {
        "type": "decay",
        "rate": 0.01,         # Tasa de decaimiento por fila
    }
}
```

**Fórmula:** `nuevo_valor = valor_antiguo * (1 - tasa) ^ indice_fila`

---

## Método: `construct_target()`

Crea o sobrescribe una variable objetivo basada en fórmulas definidas por el usuario.

### Sintaxis Básica

```python
df_with_target = scenario.construct_target(
    df=df,                              # DataFrame de entrada
    target_col="risk_score",            # Nombre columna objetivo
    formula="...",                      # Cadena o función (callable)
    noise_std=0.0,                      # Ruido Gaussiano aditivo
    task_type="regression",             # "regression" o "classification"
    threshold=None,                     # Umbral para salida binaria
)
```

### Tipos de Fórmula

#### 1. Fórmula de Cadena (String)

Expresión matemática referenciando columnas existentes:

```python
df = scenario.construct_target(
    df=df,
    target_col="risk_score",
    formula="0.3 * age + 0.5 * bmi - 0.2 * exercise_hours",
)
```

#### 2. Fórmula Llamable (Función)

Una función que toma una fila y retorna un valor:

```python
def calculate_risk(row):
    return row["age"] * 0.01 * (2 if row["smoker"] == 1 else 1)

df = scenario.construct_target(
    df=df,
    target_col="risk_score",
    formula=calculate_risk,
)
```

---

## Método: `project_to_future_period()`

Proyecta datos históricos hacia periodos futuros generando datos sintéticos y aplicando tendencias.

### Sintaxis Básica

```python
future_df = scenario.project_to_future_period(
    df=df,                              # DataFrame histórico
    periods=12,                         # Número de periodos futuros
    time_col="month",                   # Columna de tiempo
    evolution_config={...},             # Tendencias a aplicar
    generator_method="ctgan",           # Método para base sintética
    n_samples_per_period=100,           # Muestras por periodo futuro
)
```

### Flujo de Trabajo Interno

1. **Generación base sintética** usando `RealGenerator`.
2. **Asignación de periodos futuros** secuencialmente.
3. **Aplicación de `evolve_features()`** para las tendencias solicitadas.
4. **Generación de informe** comparando historial vs. proyección.

---

## Casos de Uso Exhaustivos

### Caso 1: Impacto de Recesión Económica

**Escenario:** Simular una caída del mercado donde el poder adquisitivo baja y el riesgo de impago sube.

**Solución:** Aplicar tendencia negativa a ingresos y positiva a riesgo.

```python
from calm_data_generator.generators.dynamics import ScenarioInjector

scenario = ScenarioInjector()

recession_df = scenario.evolve_features(
    df=economic_df,
    evolution_config={
        "avg_income": {"type": "trend", "slope": -50.0},  # Ingreso cae $50/día
        "unemployment_rate": {"type": "trend", "slope": 0.01}, # Desempleo sube
        "market_index": {"type": "decay", "rate": 0.005}  # Mercado colapsa exp.
    },
    time_col="date"
)

# Recalcular probabilidad de impago
recession_df = scenario.construct_target(
    df=recession_df,
    target_col="default_prob",
    # Regla: Bajo ingreso & alto desempleo = alto riesgo
    formula="0.7 * (1/avg_income) + 0.5 * unemployment_rate",
    task_type="regression"
)
```
