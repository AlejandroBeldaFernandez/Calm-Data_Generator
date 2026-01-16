# Real Generator API Reference

The `Real` generator module provides advanced tools for synthesizing data that mimics the characteristics of real-world datasets. It supports multiple synthesis methods (statistical and deep learning) and block-based generation for drift simulation.

## 1. RealGenerator

### Class Initialization
```python
class RealGenerator(
    data: pd.DataFrame,
    method: str = "cart",
    target_col: Optional[str] = None,
    block_column: Optional[str] = None,
    auto_report: bool = True,
    logger: Optional[logging.Logger] = None,
    random_state: Optional[int] = None,
    balance_target: bool = False,
    model_params: Optional[Dict[str, Any]] = None
)
```

- **Parameters:**
  - `original_data` (pd.DataFrame): The real dataset to learn from.
  - `method` (str): Synthesis method ('cart', 'rf', 'lgbm', 'gmm', 'ctgan', 'tvae', 'copula', 'datasynth', 'resample').
  - `target_column` (str, optional): Target variable name.
  - `balance_target` (bool): If True, balances the target distribution.
  - `model_params` (dict, optional): Hyperparameters for the underlying model (e.g., `{'cart_iterations': 20}`).

### `generate`
Main method to generate synthetic data.

```python
def generate(
    self,
    n_samples: int,
    output_dir: Optional[str] = None,
    drift_injection_config: Optional[List[Dict]] = None,
    date_config: Optional[DateConfig] = None,
    custom_distributions: Optional[Dict] = None,
    save_dataset: bool = False
) -> pd.DataFrame
```

- **Parameters:**
  - `n_samples` (int): Number of samples to generate.
  - `drift_injection_config` (list): List of drift inject configurations.
  - `date_config` (DateConfig): Configuration object for dates.
  - `save_dataset` (bool): If True, saves the result to CSV in `output_dir`.

---

## 2. RealBlockGenerator

### Class Initialization
```python
class RealBlockGenerator(
    original_data: pd.DataFrame,
    method: str = 'cart',
    target_column: Optional[str] = None,
    block_column: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_by_timestamp: Optional[str] = None,
    auto_report: bool = True,
    random_state: int = 42,
    verbose: bool = True,
    model_params: Optional[Dict[str, Any]] = None
)
```

- **Parameters:**
  - `block_column` (str, optional): Existing column defining blocks.
  - `chunk_size` (int, optional): Create blocks of fixed size.
  - `chunk_by_timestamp` (str, optional): Create blocks based on timestamp changes.

### `generate_block_dataset`
Generates a complete synthetic dataset by processing each block and optionally applying a drift schedule.

```python
def generate_block_dataset(
    self,
    output_dir: str,
    samples_per_block: Optional[Union[int, Dict[Any, int]]] = None,
    drift_schedule: Optional[List[Dict[str, Any]]] = None,
    custom_distributions: Optional[Dict] = None,
    date_start: Optional[str] = None,
    date_step: Optional[Dict[str, int]] = None,
    date_col: str = "timestamp"
) -> pd.DataFrame
```

- **Parameters:**
  - `samples_per_block`: Number of samples to generate per block (int or dict mapping block_id -> count).
  - `drift_schedule` (list): List of drift configurations to apply using `DriftInjector`.

### `analyze_block_statistics`
Analyzes statistics for each block in the synthetic dataset compared to the original.

```python
def analyze_block_statistics(self, synthetic_dataset: pd.DataFrame) -> Dict[str, Any]
```

### `get_block_info`
Returns detailed information about the blocks in the original dataset.

```python
def get_block_info(self) -> Dict[str, Any]
```

### `save_block_dataset`
Saves the synthetic block dataset to one or more files.

```python
def save_block_dataset(
    self,
    synthetic_dataset: pd.DataFrame,
    output_path: str,
    format: str = 'csv',
    separate_blocks: bool = False
) -> Union[str, List[str]]
```
