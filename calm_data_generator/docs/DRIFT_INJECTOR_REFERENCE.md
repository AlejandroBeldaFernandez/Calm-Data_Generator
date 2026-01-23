# DriftInjector - Complete Reference

**Location:** `calm_data_generator.generators.drift.DriftInjector`

A module to inject various types of drift (data shift) into datasets.

---

## Initialization

```python
from calm_data_generator.generators.drift import DriftInjector

injector = DriftInjector(
    output_dir="./drift_output",      # Output directory for results
    generator_name="my_drift",        # Base name for output files
    random_state=42,                  # Seed for reproducibility
    time_col="timestamp",             # Default time column
    block_column="block",             # Default block column
    target_column="target",           # Default target column
    auto_report=True,                 # Generate reports automatically
    minimal_report=False,             # Full reports
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"drift_output"` | Directory for outputs |
| `generator_name` | str | `"DriftInjector"` | Base name for files |
| `random_state` | int | `None` | Seed for reproducibility |
| `time_col` | str | `None` | Default time column |
| `block_column` | str | `None` | Default block column |
| `target_column` | str | `None` | Default target column |
| `auto_report` | bool | `True` | Automatically generate a quality report |
| `minimal_report` | bool | `False` | Simplified reports |

---

## Operation Types (`drift_type`)

For **numeric** columns:

| Type | Description | Formula |
|------|-------------|---------|
| `gaussian_noise` | Gaussian noise | `x + N(0, magnitude * std)` |
| `uniform_noise` | Uniform noise | `x + U(-mag*std, +mag*std)` |
| `shift` | Mean shift | `x + (mean * magnitude)` |
| `scale` | Scale around mean | `mean + (x - mean) * (1 + magnitude)` |
| `add_value` | Add fixed value | `x + drift_value` |
| `subtract_value` | Subtract fixed value | `x - drift_value` |
| `multiply_value` | Multiply by factor | `x * drift_value` |
| `divide_value` | Divide by factor | `x / drift_value` |

For **categorical** columns: Random replacement by another existing category.

---

## Row Selection

All methods support multiple ways to select which rows to affect:

### By Index
```python
injector.inject_feature_drift(
    ...,
    start_index=100,      # From row 100
    end_index=500,        # To row 500
    index_step=2,         # Every 2 rows
)
```

### By Blocks
```python
# Specific blocks
injector.inject_feature_drift(..., block_column="hospital", blocks=["H1", "H2"])

# Block range
injector.inject_feature_drift(..., block_start="H1", n_blocks=3, block_step=1)
```

### By Time
```python
# Time range
injector.inject_feature_drift(
    ...,
    time_col="timestamp",
    time_start="2024-01-01",
    time_end="2024-06-30",
)

# Multiple ranges
injector.inject_feature_drift(
    ...,
    time_ranges=[
        ("2024-01-01", "2024-03-31"),
        ("2024-07-01", "2024-09-30"),
    ],
)

# Specific timestamps
injector.inject_feature_drift(
    ...,
    specific_times=["2024-01-15", "2024-02-15", "2024-03-15"],
)
```

---

## Feature Drift Methods

### `inject_feature_drift()` - Abrupt Shift

```python
drifted_df = injector.inject_feature_drift(
    df=df,
    feature_cols=["price", "quantity"],
    drift_type="shift",
    drift_magnitude=0.3,
    drift_value=10.0,
    start_index=500,
)
```

### `inject_feature_drift_gradual()` - Soft Transition

```python
drifted_df = injector.inject_feature_drift_gradual(
    df=df,
    feature_cols=["price"],
    drift_type="shift",
    drift_magnitude=0.5,
    center=500,              # Row index of transition center
    width=200,               # Width of transition window
    profile="sigmoid",       # Transition profile: "sigmoid", "linear", "cosine"
    speed_k=1.0,             # Slope of the transition (higher = steeper)
    direction="up",          # "up" (0→1) or "down" (1→0)
    inconsistency=0.0,       # Noise in transition (0.0 to 1.0)
)
```

**Transition Profiles:**

| Profile | Shape | Usage |
|---------|-------|-------|
| `sigmoid` | S-curve | Natural transitions |
| `linear` | Straight ramp | Constant change |
| `cosine` | Smooth cosine | Organic transitions |

### `inject_feature_drift_incremental()` - Continuous Linear Growth

```python
drifted_df = injector.inject_feature_drift_incremental(
    df=df,
    feature_cols=["score"],
    drift_type="shift",
    drift_magnitude=0.5,
    start_index=0,
    end_index=1000,
)
```

### `inject_feature_drift_recurrent()` - Cycles

```python
drifted_df = injector.inject_feature_drift_recurrent(
    df=df,
    feature_cols=["value"],
    drift_type="scale",
    drift_magnitude=0.3,
    repeats=4,                    # Number of cycles
    random_repeat_order=False,    # Randomize repeat order
)
```

---

## Conditional Drift

### `inject_conditional_drift()` - Rule-based Drift

Injects drift (abrupt, gradual, incremental, or recurrent) on a subset of data defined by conditions.

```python
drifted_df = injector.inject_conditional_drift(
    df=df,
    feature_cols=["salary"],
    conditions=[
        {"column": "age", "operator": ">", "value": 50},
        {"column": "city", "operator": "==", "value": "New York"}
    ],
    drift_type="scale",
    drift_magnitude=0.3,
    drift_method="gradual",  # "abrupt", "gradual", "incremental", "recurrent"
    center=100,
    width=50,
)
```

**Supported Operators:**

| Operator | Description | Example |
|----------|-------------|---------|
| `>` | Greater than | `{"column": "age", "operator": ">", "value": 30}` |
| `>=` | Greater or equal | `{"column": "score", "operator": ">=", "value": 0.5}` |
| `<` | Less than | `{"column": "price", "operator": "<", "value": 100}` |
| `<=` | Less or equal | `{"column": "qty", "operator": "<=", "value": 10}` |
| `==` | Equals to | `{"column": "city", "operator": "==", "value": "Madrid"}` |
| `!=` | Not equal | `{"column": "status", "operator": "!=", "value": "inactive"}` |
| `in` | In list | `{"column": "type", "operator": "in", "value": ["A", "B"]}` |

---

## Label Drift (Target)

### `inject_label_drift()` - Random Flip

```python
drifted_df = injector.inject_label_drift(
    df=df,
    target_cols=["fraud"],
    drift_magnitude=0.2,              # 20% of labels flipped
    start_index=500,
)
```

### `inject_label_shift()` - Force Distribution

```python
drifted_df = injector.inject_label_shift(
    df=df,
    target_col="Churn",
    target_distribution={0: 0.9, 1: 0.1},   # 90% class 0, 10% class 1
    start_index=500,
)
```

---

## Categorical Drift

### `inject_categorical_frequency_drift()` - Probability Change

```python
drifted_df = injector.inject_categorical_frequency_drift(
    df=df,
    feature_cols=["payment_method"],
    drift_magnitude=0.5,        # 50% of rows resampled
    perturbation="invert",      # "uniform", "invert", "random"
)
```

**Perturbation Strategies:**

- `uniform`: Tends towards uniform distribution (maximum entropy).
- `invert`: Inverts frequencies (rare categories become common).
- `random`: Random distribution.

### `inject_typos_drift()` - Text Noise

```python
drifted_df = injector.inject_typos_drift(
    df=df,
    feature_cols=["city"],
    drift_magnitude=0.1,        # 10% of rows affected
    typo_density=1,             # Typos per string
    typo_type="random",         # "swap", "delete", "duplicate", "random"
)
```

### `inject_category_merge_drift()` - Merge Categories

```python
drifted_df = injector.inject_category_merge_drift(
    df=df,
    col="vehicle_type",
    categories_to_merge=["Car", "SUV"],
    new_category_name="Luxury Vehicle",
)
```

---

## Boolean Drift

```python
drifted_df = injector.inject_boolean_drift(
    df=df,
    feature_cols=["is_active"],
    drift_magnitude=0.3,        # 30% of values flipped
)
```

---

## Data Quality Issues

| Method | Description |
|--------|-------------|
| `inject_nulls` | Injects MCAR missing values |
| `inject_outliers_global` | Global outliers scaled by a factor |
| `inject_correlation_matrix_drift` | Covariate drift by changing feature correlation structure |

---

## Summary of Methods

| Category | Method | Description |
|----------|--------|-------------|
| **Features** | `inject_feature_drift` | Abrupt change |
| | `inject_feature_drift_gradual` | Soft transition |
| | `inject_feature_drift_incremental` | Linear growth |
| | `inject_feature_drift_recurrent` | Periodical cycles |
| **Conditional** | `inject_conditional_drift` | Rule-based |
| **Labels** | `inject_label_drift` | Target flip |
| | `inject_label_shift` | Probability shift |
| **Categorical** | `inject_categorical_frequency_drift` | Proportions change |
| | `inject_typos_drift` | Text errors |
| | `inject_category_merge_drift` | Grouping categories |
| | `inject_new_value` | Missing category injection |
| **Boolean** | `inject_boolean_drift` | Logical flip |
| **Quality** | `inject_nulls` | Missing values |
| | `inject_outliers_global` | Extreme values |
| | `inject_correlation_matrix_drift` | Correlation shift |
| **Orchestration** | `inject_multiple_types_of_drift` | Multi-method schedule |

---

## Comprehensive Use Cases

### Case 1: Sensor Degradation Simulation

**Scenario:** Simulating an IoT sensor that gradually loses calibration (shifts) and becomes noisier over time.

**Solution:** Combine incremental shift with increasing noise.

```python
from calm_data_generator.generators.drift import DriftInjector

injector = DriftInjector()

# Step 1: Linear drift (calibration loss)
df_drifted = injector.inject_feature_drift_incremental(
    df=sensor_df,
    feature_cols=["sensor_reading"],
    drift_type="shift",
    drift_magnitude=0.5,  # Shift mean by 50% by the end
)

# Step 2: Increasing noise (component wear)
# We can simulate this by splitting data into chunks and applying higher noise to later chunks
df_drifted = injector.inject_feature_drift(
    df=df_drifted,
    feature_cols=["sensor_reading"],
    drift_type="gaussian_noise",
    drift_magnitude=0.1,  # Initial noise
    start_index=0,
    end_index=500
)

df_drifted = injector.inject_feature_drift(
    df=df_drifted,
    feature_cols=["sensor_reading"],
    drift_type="gaussian_noise",
    drift_magnitude=0.3,  # High noise
    start_index=500
)
```

### Case 2: Seasonal Pattern Injection

**Scenario:** Adding a simulated holiday season effect where sales increase periodically.

**Solution:** Use `recurrent` drift.

```python
sales_drifted = injector.inject_feature_drift_recurrent(
    df=sales_df,
    feature_cols=["daily_sales"],
    drift_type="multiply_value",
    drift_magnitude=1.5,  # 50% increase during "season"
    repeats=3,            # 3 holiday seasons in the dataset
    random_repeat_order=False
)
```

### Case 3: Concept Drift (Relationship Change)

**Scenario:** The relationship between features (age, income) and target (loan default) changes. Younger people with high income suddenly start defaulting more.

**Solution:** Use `conditional` label drift.

```python
# Introduce drift: Young rich people default more (flip label to 1)
df_concept_drift = injector.inject_conditional_drift(
    df=loan_df,
    feature_cols=["default"],   # Target column
    conditions=[
        {"column": "age", "operator": "<", "value": 30},
        {"column": "income", "operator": ">", "value": 80000}
    ],
    drift_type="add_value",     # Assuming default is 0/1, adding may flip 0->1
    drift_magnitude=1.0,        # Force change
    drift_method="gradual",     # Change happens slowly
    center=1000
)
```

### Case 4: Data Quality Stress Test (Broken Pipeline)

**Scenario:** Simulating a pipeline failure where a categorical column starts receiving "ERROR" values or NULLs.

**Solution:** Use category injection or null injection.

```python
# Inject 'ERROR' category
df_broken = injector.inject_new_category_drift(
    df=log_df,
    feature_col="status",
    new_category="ERROR_TIMEOUT",
    candidate_logic={"percentage": 0.2}, # 20% of rows become errors
    start_index=800
)

# Inject random NULLs
df_broken = injector.inject_nulls(
    df=df_broken,
    feature_cols=["response_time"],
    missing_fraction=0.15,
    start_index=800
)
```
