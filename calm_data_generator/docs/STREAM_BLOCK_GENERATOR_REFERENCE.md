# Stream Block Generator Reference

The `calm_data_generator.generators.stream.StreamBlockGenerator` module provides the `SyntheticBlockGenerator` class, which simplifies the creation of synthetic datasets composed of multiple distinct blocks. This is particularly useful for simulating non-stationary environments and concept drift by using different generator parameters for different blocks.

## Class: `SyntheticBlockGenerator`

### Usage
```python
from calm_data_generator.generators.stream.StreamBlockGenerator import SyntheticBlockGenerator

block_gen = SyntheticBlockGenerator()

# Generate 3 blocks of data using the 'sea' concept
path = block_gen.generate_blocks_simple(
    output_dir="./output_stream_blocks",
    filename="stream_blocks.csv",
    n_blocks=3,
    total_samples=3000,
    methods="sea",
    method_params=[
         {"variant": 0}, # Block 1
         {"variant": 1}, # Block 2 (concept drift)
         {"variant": 2}  # Block 3 (concept drift)
    ]
)
```

### `generate_blocks_simple`
**Signature:** `generate_blocks_simple(...)`

A simplified interface for generating block-structured datasets using string-based method names.

- **Args:**
    - `output_dir` (str): Output directory.
    - `filename` (str): Output CSV filename.
    - `n_blocks` (int): Number of blocks.
    - `total_samples` (int): Total samples across all blocks.
    - `methods` (Union[str, List[str]]): Generator method name(s) (e.g., 'sea', 'agrawal', 'hyperplane').
    - `method_params` (Union[Dict, List[Dict]]): Parameters for the generator(s) per block.
    - `n_samples_block`: Override for samples per block.
    - `drift_config`: List of drift configurations to apply.
    - `dynamics_config`: Configuration for dynamics injection (feature evolution, etc.).

### `generate`
**Signature:** `generate(...)`

Generates a block-structured dataset from a list of instantiated River generator objects.

- **Args:**
    - `generators` (List): List of instantiated River generator objects (one per block).
    - `n_blocks`: Number of blocks.
    - `n_samples_block`: List of sample counts per block.
    - `block_labels`: Optional list of labels for the blocks.
    - `date_start`, `date_step`, `date_col`: Date injection settings.
    - `generate_report` (bool): Whether to generate a comprehensive report.

- **Returns:** `str`: Full path to the generated CSV file.
