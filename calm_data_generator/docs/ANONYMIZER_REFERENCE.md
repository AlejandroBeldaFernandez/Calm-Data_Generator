# Anonymizer Reference

The `calm_data_generator.anonymizer` module provides utilities for privacy-preserving data transformations. These functions help in anonymizing sensitive data to comply with privacy regulations like GDPR or HIPAA.

## Module: `calm_data_generator.anonymizer.privacy`

### `pseudonymize_columns`
**Signature:** `pseudonymize_columns(df: pd.DataFrame, columns: list, salt: str = None) -> pd.DataFrame`

Pseudonymizes specified columns in a DataFrame using SHA256 hashing.

- **Args:**
    - `df`: The input DataFrame.
    - `columns`: List of column names to pseudonymize.
    - `salt`: Optional salt string to add to the hashing process for security.

- **Returns:** A new DataFrame with specified columns pseudonymized.

### `add_laplace_noise`
**Signature:** `add_laplace_noise(df: pd.DataFrame, columns: list, epsilon: float = 1.0) -> pd.DataFrame`

Applies Laplace noise to specified numeric columns for differential privacy.

- **Args:**
    - `df`: The input DataFrame.
    - `columns`: List of numeric column names to add noise to.
    - `epsilon`: The privacy budget (smaller value = more privacy & more noise). Defaults to 1.0.

- **Returns:** A new DataFrame with noise added to specified columns.

### `generalize_numeric_to_ranges`
**Signature:** `generalize_numeric_to_ranges(df: pd.DataFrame, columns: list, num_bins: int = 5) -> pd.DataFrame`

Generalizes specified numeric columns by binning their values into ranges (k-anonymity technique).

- **Args:**
    - `df`: The input DataFrame.
    - `columns`: List of numeric column names to generalize.
    - `num_bins`: Number of bins/ranges to create. Defaults to 5.

- **Returns:** A new DataFrame with specified columns generalized into string-based ranges.

### `generalize_categorical_by_mapping`
**Signature:** `generalize_categorical_by_mapping(df: pd.DataFrame, columns: list, mapping: dict) -> pd.DataFrame`

Generalizes specified categorical columns by applying a user-defined mapping.

- **Args:**
    - `df`: The input DataFrame.
    - `columns`: List of categorical column names to generalize.
    - `mapping`: A dictionary defining the map from old values to new, generalized values.

- **Returns:** A new DataFrame with specified columns altered.

### `shuffle_columns`
**Signature:** `shuffle_columns(df: pd.DataFrame, columns: list, random_state: int = None) -> pd.DataFrame`

Shuffles values within specified columns independently to break correlations while preserving column distributions.

- **Args:**
    - `df`: The input DataFrame.
    - `columns`: List of column names to shuffle.
    - `random_state`: Seed for reproducibility.

- **Returns:** A new DataFrame with specified columns shuffled.
