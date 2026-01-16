# Drift Injector API Reference

The `DriftInjection` module provides a comprehensive toolset for injecting various types of drift (feature, label, concept, virtual) into real datasets. It supports complex drift profiles like gradual, abrupt, incremental, and recurrent drifts, with flexible targeting by index, block, or time.

## DriftInjector

### Class Initialization
```python
class DriftInjector(
    original_df: pd.DataFrame,
    output_dir: str,
    generator_name: str,
    target_column: Optional[str] = None,
    block_column: Optional[str] = None,
    time_col: Optional[str] = None,
    random_state: Optional[int] = None
)
```

- **Parameters:**
  - `original_df` (pd.DataFrame): The original dataset.
  - `output_dir` (str): Directory for reports and output.
  - `generator_name` (str): Name used for output files.
  - `target_column` (str, optional): Target variable name.
  - `block_column` (str, optional): Column defining data blocks.
  - `time_col` (str, optional): Timestamp column name.

---

## Feature Drift Methods

### `inject_feature_drift`
Applies drift at once to selected rows.

```python
def inject_feature_drift(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    drift_type: str = "gaussian_noise",
    drift_magnitude: float = 0.2,
    drift_value: Optional[float] = None,
    drift_values: Optional[Dict[str, float]] = None,
    start_index: Optional[int] = None,
    block_index: Optional[int] = None,
    block_column: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
    specific_times: Optional[Sequence[str]] = None,
    auto_report: bool = True
) -> pd.DataFrame
```

- **Parameters:**
  - `drift_type` (str): 'gaussian_noise', 'shift', 'scale', 'add_value', 'subtract_value', 'multiply_value', 'divide_value'.
  - `drift_magnitude` (float): Magnitude of the drift.

### `inject_feature_drift_gradual`
Injects gradual drift using a transition window (sigmoid, linear, cosine).

```python
def inject_feature_drift_gradual(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    drift_type: str = "gaussian_noise",
    drift_magnitude: float = 0.2,
    center: Optional[int] = None,
    width: Optional[int] = None,
    profile: str = "sigmoid",
    speed_k: float = 1.0,
    direction: str = "up",
    inconsistency: float = 0.0,
    **kwargs
) -> pd.DataFrame
```

### `inject_feature_drift_abrupt`
Injects abrupt drift (fast sigmoid transition).

```python
def inject_feature_drift_abrupt(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    drift_type: str,
    drift_magnitude: float,
    change_index: Optional[int] = None,
    width: int = 3,
    **kwargs
) -> pd.DataFrame
```

### `inject_feature_drift_incremental`
Injects constant and smooth drift over the selected range.

```python
def inject_feature_drift_incremental(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    drift_type: str,
    drift_magnitude: float,
    **kwargs
) -> pd.DataFrame
```

### `inject_feature_drift_recurrent`
Injects recurrent drift by applying multiple drift windows.

```python
def inject_feature_drift_recurrent(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    drift_type: str,
    drift_magnitude: float,
    windows: Optional[Sequence[Tuple[int, int]]] = None,
    repeats: int = 1,
    **kwargs
) -> pd.DataFrame
```

### `inject_conditional_drift`
Injects drift on a subset of data based on specific conditions.

```python
def inject_conditional_drift(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    conditions: List[Dict[str, Any]],
    drift_type: str,
    drift_magnitude: float,
    **kwargs
) -> pd.DataFrame
```

- **Parameters:**
  - `conditions` (list): List of dicts, e.g., `[{"column": "age", "operator": ">", "value": 50}]`.

---

## Label Drift Methods

### `inject_label_drift`
Injects random label flips.

```python
def inject_label_drift(
    self,
    df: pd.DataFrame,
    target_cols: List[str],
    drift_magnitude: float = 0.1,
    **kwargs
) -> pd.DataFrame
```

### `inject_label_drift_gradual`
Injects gradual label drift.

```python
def inject_label_drift_gradual(
    self,
    df: pd.DataFrame,
    target_col: str,
    drift_magnitude: float = 0.3,
    center: Optional[int] = None,
    width: Optional[int] = None,
    **kwargs
) -> pd.DataFrame
```

### `inject_label_drift_abrupt`
Injects abrupt label drift.

```python
def inject_label_drift_abrupt(
    self,
    df: pd.DataFrame,
    target_col: str,
    drift_magnitude: float,
    change_index: int,
    **kwargs
) -> pd.DataFrame
```

### `inject_label_shift`
Injects label shift by resampling the target column to match a new distribution.

```python
def inject_label_shift(
    self,
    df: pd.DataFrame,
    target_col: str,
    target_distribution: dict,
    **kwargs
) -> pd.DataFrame
```

---

## Other Drift Methods

### `inject_missing_values_drift`
Injects missing values (NaN) into specified columns.

```python
def inject_missing_values_drift(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    missing_fraction: float = 0.1,
    **kwargs
) -> pd.DataFrame
```

### `inject_correlation_matrix_drift`
Injects covariate drift by transforming numeric features to match a new correlation matrix.

```python
def inject_correlation_matrix_drift(
    self,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_correlation_matrix: np.ndarray,
    **kwargs
) -> pd.DataFrame
```

### `inject_new_category_drift`
Injects a new category into a feature column.

```python
def inject_new_category_drift(
    self,
    df: pd.DataFrame,
    feature_col: str,
    new_category: object,
    candidate_logic: dict,
    **kwargs
) -> pd.DataFrame
```
