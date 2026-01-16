# Privacy Module API Reference

The `Privacy` module provides functions for data anonymization, differential privacy, and generalization.

## Functions

### `pseudonymize_columns`
Pseudonymizes specified columns using SHA256 hashing.

```python
def pseudonymize_columns(
    df: pd.DataFrame,
    columns: list,
    salt: str = None
) -> pd.DataFrame
```

- **Parameters:**
  - `df` (pd.DataFrame): The input DataFrame.
  - `columns` (list): List of column names to pseudonymize.
  - `salt` (str, optional): Salt string to add to the hash for security.

- **Returns:**
  - `pd.DataFrame`: DataFrame with hashed columns.

---

### `add_laplace_noise`
Applies Laplace noise to numeric columns for differential privacy.

```python
def add_laplace_noise(
    df: pd.DataFrame,
    columns: list,
    epsilon: float = 1.0
) -> pd.DataFrame
```

- **Parameters:**
  - `df` (pd.DataFrame): The input DataFrame.
  - `columns` (list): List of numeric columns.
  - `epsilon` (float): Privacy budget. Smaller values = more noise/privacy. Default 1.0.

- **Returns:**
  - `pd.DataFrame`: DataFrame with noisy numeric columns.

---

### `generalize_numeric_to_ranges`
Generalizes numeric columns by binning values into ranges (k-anonymity).

```python
def generalize_numeric_to_ranges(
    df: pd.DataFrame,
    columns: list,
    num_bins: int = 5
) -> pd.DataFrame
```

- **Parameters:**
  - `df` (pd.DataFrame): The input DataFrame.
  - `columns` (list): List of numeric columns.
  - `num_bins` (int): Number of bins to create. Default 5.

- **Returns:**
  - `pd.DataFrame`: DataFrame with columns converted to string ranges.

---

### `generalize_categorical_by_mapping`
Generalizes categorical columns using a custom mapping.

```python
def generalize_categorical_by_mapping(
    df: pd.DataFrame,
    columns: list,
    mapping: dict
) -> pd.DataFrame
```

- **Parameters:**
  - `df` (pd.DataFrame): The input DataFrame.
  - `columns` (list): List of categorical columns.
  - `mapping` (dict): Dictionary mapping old values to new generalized values.

- **Returns:**
  - `pd.DataFrame`: DataFrame with mapped values.

---

### `shuffle_columns`
Shuffles values within specified columns to break correlations.

```python
def shuffle_columns(
    df: pd.DataFrame,
    columns: list,
    random_state: int = None
) -> pd.DataFrame
```

- **Parameters:**
  - `df` (pd.DataFrame): The input DataFrame.
  - `columns` (list): List of columns to shuffle.
  - `random_state` (int, optional): Seed for reproducibility.

- **Returns:**
  - `pd.DataFrame`: DataFrame with shuffled columns.
