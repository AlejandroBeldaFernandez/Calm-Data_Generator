# Referencia del Stream Block Generator

El módulo `calm_data_generator.generators.stream.StreamBlockGenerator` proporciona la clase `SyntheticBlockGenerator`, que simplifica la creación de datasets sintéticos compuestos por múltiples bloques distintos. Esto es particularmente útil para simular entornos no estacionarios y concept drift usando diferentes parámetros de generador para diferentes bloques.

## Clase: `SyntheticBlockGenerator`

### Uso
```python
from calm_data_generator.generators.stream.StreamBlockGenerator import SyntheticBlockGenerator

block_gen = SyntheticBlockGenerator()

# Generar 3 bloques de datos usando el concepto 'sea'
path = block_gen.generate_blocks_simple(
    output_dir="./output_stream_blocks",
    filename="stream_blocks.csv",
    n_blocks=3,
    total_samples=3000,
    methods="sea",
    method_params=[
         {"variant": 0}, # Bloque 1
         {"variant": 1}, # Bloque 2 (concept drift)
         {"variant": 2}  # Bloque 3 (concept drift)
    ]
)
```

### `generate_blocks_simple`
**Firma:** `generate_blocks_simple(...)`

Una interfaz simplificada para generar datasets estructurados en bloques usando nombres de métodos basados en strings.

- **Argumentos:**
    - `output_dir` (str): Directorio de salida.
    - `filename` (str): Nombre archivo CSV de salida.
    - `n_blocks` (int): Número de bloques.
    - `total_samples` (int): Muestras totales entre todos los bloques.
    - `methods` (Union[str, List[str]]): Nombre(s) de método generador (ej., 'sea', 'agrawal', 'hyperplane').
    - `method_params` (Union[Dict, List[Dict]]): Parámetros para el/los generador(es) por bloque.
    - `n_samples_block`: Sobrescribir muestras por bloque.
    - `drift_config`: Lista de configuraciones de drift a aplicar.
    - `dynamics_config`: Configuración para inyección de dinámicas (evolución de features, etc.).

### `generate`
**Firma:** `generate(...)`

Genera un dataset estructurado en bloques a partir de una lista de objetos generadores River instanciados.

- **Argumentos:**
    - `generators` (List): Lista de objetos generadores River instanciados (uno por bloque).
    - `n_blocks`: Número de bloques.
    - `n_samples_block`: Lista de conteo de muestras por bloque.
    - `block_labels`: Lista opcional de etiquetas para los bloques.
    - `date_start`, `date_step`, `date_col`: Configuración de inyección de fechas.
    - `generate_report` (bool): Si generar un informe completo.

- **Retorna:** `str`: Ruta completa al archivo CSV generado.
