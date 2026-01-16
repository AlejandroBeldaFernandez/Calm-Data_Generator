# `StreamGenerator` Documentation

The `Synthetic` generator module provides tools to generate synthetic tabular datasets (batch) by consuming data streams from the `river` library. It supports simulating various types of concept drift.

## Generators

### 1. `StreamGenerator`
A wrapper around `river` datasets (e.g., SEA, AGRAWAL, STAGGER) that consumes the stream to generate a static dataset (pandas DataFrame or CSV) with configurable drift (abrupt, gradual, etc.).

### 2. `SyntheticBlockGenerator`
A high-level abstraction for generating block-structured datasets. It allows:
- Defining a sequence of blocks.
- Applying different generation methods or parameters per block.
- Simulating complex drift scenarios over time.

## Installation

Ensure `calmops` and `river` are installed.

## Basic Usage (`StreamGenerator`)

```python
from calmops.data_generators.Synthetic.StreamGenerator import StreamGenerator
from river.datasets import synth

# 1. Initialize Generator (Lightweight)
generator = StreamGenerator(seed=42)

# 2. Define Stream Source (River)
sea_stream = synth.SEA(variant=0)

# 3. Generate Data
data = generator.generate(
    generator_instance=sea_stream,
    n_samples=1000
)
```

## Block Usage (`SyntheticBlockGenerator`)

```python
from calmops.data_generators.Synthetic.SyntheticBlockGenerator import SyntheticBlockGenerator

block_gen = SyntheticBlockGenerator()

# Generate blocks with different concepts
block_gen.generate_blocks_simple(
    output_path="synthetic_blocks",
    filename="data.csv",
    n_blocks=3,
    total_samples=3000,
    methods="sea",
    method_params=[
        {"function": 1}, # Block 1: Concept 1
        {"function": 2}, # Block 2: Concept 2 (Abrupt Drift)
        {"function": 1}  # Block 3: Concept 1 (Recurrent Drift)
    ]
)
```

## Tutorial

For a complete example, run the `tutorial.py` script in this directory:

```bash
python tutorial.py
```
