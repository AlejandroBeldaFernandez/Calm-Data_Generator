# Referencia del Anonimizador

El módulo `calm_data_generator.anonymizer` proporciona utilidades para transformaciones de datos que preservan la privacidad. Estas funciones ayudan a anonimizar datos sensibles para cumplir con regulaciones de privacidad como GDPR o HIPAA.

## Módulo: `calm_data_generator.anonymizer.privacy`

### `pseudonymize_columns`
**Firma:** `pseudonymize_columns(df: pd.DataFrame, columns: list, salt: str = None) -> pd.DataFrame`

Pseudonimiza las columnas especificadas en un DataFrame usando hashing SHA256.

- **Argumentos:**
    - `df`: El DataFrame de entrada.
    - `columns`: Lista de nombres de columnas a pseudonimizar.
    - `salt`: String de salt opcional para añadir al proceso de hashing para mayor seguridad.

- **Retorna:** Un nuevo DataFrame con las columnas especificadas pseudonimizadas.

### `add_laplace_noise`
**Firma:** `add_laplace_noise(df: pd.DataFrame, columns: list, epsilon: float = 1.0) -> pd.DataFrame`

Aplica ruido de Laplace a columnas numéricas especificadas para privacidad diferencial.

- **Argumentos:**
    - `df`: El DataFrame de entrada.
    - `columns`: Lista de nombres de columnas numéricas a las que añadir ruido.
    - `epsilon`: El presupuesto de privacidad (valor menor = más privacidad y más ruido). Por defecto 1.0.

- **Retorna:** Un nuevo DataFrame con ruido añadido a las columnas especificadas.

### `generalize_numeric_to_ranges`
**Firma:** `generalize_numeric_to_ranges(df: pd.DataFrame, columns: list, num_bins: int = 5) -> pd.DataFrame`

Generaliza columnas numéricas especificadas agrupando sus valores en rangos (técnica de k-anonimidad).

- **Argumentos:**
    - `df`: El DataFrame de entrada.
    - `columns`: Lista de nombres de columnas numéricas a generalizar.
    - `num_bins`: Número de contenedores/rangos a crear (bins). Por defecto 5.

- **Retorna:** Un nuevo DataFrame con las columnas especificadas generalizadas en rangos basados en texto.

### `generalize_categorical_by_mapping`
**Firma:** `generalize_categorical_by_mapping(df: pd.DataFrame, columns: list, mapping: dict) -> pd.DataFrame`

Generaliza columnas categóricas especificadas aplicando un mapeo definido por el usuario.

- **Argumentos:**
    - `df`: El DataFrame de entrada.
    - `columns`: Lista de nombres de columnas categóricas a generalizar.
    - `mapping`: Un diccionario definiendo el mapa de valores antiguos a nuevos valores generalizados.

- **Retorna:** Un nuevo DataFrame con las columnas especificadas alteradas.

### `shuffle_columns`
**Firma:** `shuffle_columns(df: pd.DataFrame, columns: list, random_state: int = None) -> pd.DataFrame`

Baraja los valores dentro de las columnas especificadas independientemente para romper correlaciones mientras preserva las distribuciones de las columnas.

- **Argumentos:**
    - `df`: El DataFrame de entrada.
    - `columns`: Lista de nombres de columnas a barajar.
    - `random_state`: Semilla para reproducibilidad.

- **Retorna:** Un nuevo DataFrame con las columnas especificadas barajadas.
