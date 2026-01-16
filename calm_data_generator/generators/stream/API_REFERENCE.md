# Synthetic Generator API Reference

The `Synthetic` module provides tools for generating synthetic tabular datasets based on data streams from the `river` library. It supports simulating various types of concept drift.

## 1. StreamGenerator

### Class Initialization
```python
class StreamGenerator(random_state: Optional[int] = None)
```
- **Parameters:**
  - `random_state` (int, optional): Seed for reproducibility.

### `generate`
Main method to generate a synthetic dataset from a River generator.

```python
def generate(
    self,
    generator_instance,
    n_samples: int,
    output_dir: Optional[str] = None,
    drift_injection_config: Optional[List[Dict]] = None,
    date_config: Optional[DateConfig] = None,
    drift_type: str = "none",
    generator_instance_drift: Optional[object] = None,
    drift_point: int = None,
    transition_width: int = None,
    target_col: str = "target",
    balance: bool = False,
    save_dataset: bool = True,
    generate_report: bool = True,
    metadata_generator_instance: Optional[object] = None,
    **kwargs
) -> pd.DataFrame
```

- **Parameters:**
  - `generator_instance`: An instantiated River generator (e.g., `river.datasets.synth.SEA()`).
  - `n_samples` (int): Number of samples to generate.
  - `drift_type` (str): Type of drift ('none', 'virtual', 'gradual', 'incremental', 'abrupt').
  - `drift_generator` (object): A second River generator instance required for drift types other than 'none' or 'virtual'.
  - `position_of_drift` (int): Sample index where drift starts.
  - `transition_width` (int): Width of the drift transition (for gradual drift).
  - `balance` (bool): If True, balances class distribution.
  - `date_start` (str): Start date for timestamp injection (e.g., "2024-01-01").

---

## 2. SyntheticBlockGenerator

### Class Initialization
```python
class SyntheticBlockGenerator()
```

### `generate_blocks_simple`
Simplified interface for generating block-structured datasets using string-based method names.

```python
def generate_blocks_simple(
    self,
    output_path: str,
    filename: str,
    n_blocks: int,
    total_samples: int,
    methods: Union[str, List[str]],
    method_params: Union[Dict, List[Dict]] = None,
    instances_per_block: Union[int, List[int]] = None,
    target_col: str = "target",
    balance: bool = False,
    random_state: int = None,
    date_start: str = None,
    date_step: dict = None,
    date_col: str = "timestamp",
    generate_report: bool = True
)
```

- **Parameters:**
  - `methods` (str or list): Name of generator method (e.g., "sea", "agrawal").
  - `method_params` (dict or list): Parameters for the generator (e.g., `{'function': 1}`).
  - `n_blocks` (int): Number of blocks to generate.
  - `total_samples` (int): Total samples across all blocks.

### `generate_blocks`
Advanced interface that accepts instantiated River generator objects.

```python
def generate_blocks(
    self,
    output_path: str,
    filename: str,
    n_blocks: int,
    total_samples: int,
    instances_per_block: List[int],
    generators: List,
    target_col: str = "target",
    balance: bool = False,
    date_start: str = None,
    date_step: dict = None,
    date_col: str = "timestamp",
    generate_report: bool = True
) -> str
```
