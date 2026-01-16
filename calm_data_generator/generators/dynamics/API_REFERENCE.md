# Dynamics Injector API Reference

The `Dynamics` module provides a standalone tool for evolving features over time and constructing target variables based on dynamic formulas. It is useful for simulating feature drift and concept drift in existing datasets.

## DynamicsInjector

### Class Initialization
```python
class DynamicsInjector(seed: Optional[int] = None)
```
- **Parameters:**
  - `seed` (int, optional): Random seed for reproducibility.

### `evolve_features`
Evolves features in a DataFrame based on a configuration dictionary.

```python
def evolve_features(
    self,
    df: pd.DataFrame,
    evolution_config: Dict[str, Dict[str, Union[str, float, int]]],
    time_col: Optional[str] = None
) -> pd.DataFrame
```

- **Parameters:**
  - `df` (pd.DataFrame): Input DataFrame.
  - `evolution_config` (dict): Dictionary mapping column names to evolution specs.
    - Supported types: `'linear'`, `'cycle'` (or `'sinusoidal'`), `'sigmoid'`.
    - Example: `{'Age': {'type': 'linear', 'slope': 0.1}}`
  - `time_col` (str, optional): Column to use as the time variable `t`. If None, uses the index.

- **Returns:**
  - `pd.DataFrame`: DataFrame with evolved features.

### `construct_target`
Constructs or overwrites a target variable based on a formula.

```python
def construct_target(
    self,
    df: pd.DataFrame,
    target_col: str,
    formula: Union[str, Callable],
    noise_std: float = 0.0,
    task_type: str = "regression",
    threshold: Optional[float] = None
) -> pd.DataFrame
```

- **Parameters:**
  - `target_col` (str): Name of the target column.
  - `formula` (str or callable): Formula to calculate the raw target score.
    - String: Used in `df.eval()`.
    - Callable: Accepts `df` and returns a Series/array.
  - `noise_std` (float): Standard deviation of Gaussian noise to add.
  - `task_type` (str): `'regression'` or `'classification'`.
  - `threshold` (float, optional): Threshold for binary classification (if `task_type='classification'`).

- **Returns:**
  - `pd.DataFrame`: DataFrame with the new target column.
