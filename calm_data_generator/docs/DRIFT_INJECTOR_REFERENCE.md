# DriftInjector - Complete Reference

**Location:** `calm_data_generator.injectors.DriftInjector`

Tool for injecting realistic data drift patterns into synthetic datasets.

---

## Quick Start Guide

### What is DriftInjector?

A tool to simulate **data drift** - changes in data distribution over time. Essential for testing:
- 🔄 **Model monitoring** systems
- �� **Drift detection** algorithms
- 🎯 **Adaptive ML** pipelines
- ⚠️ **Alert systems** for production models

### When to Use Each Drift Type

| Drift Type | What It Does | When to Use |
|------------|--------------|-------------|
| **Feature Drift (Gradual)** | Slowly shifts feature values | Seasonal changes, aging populations |
| **Feature Drift (Sudden)** | Abrupt feature change | System updates, policy changes |
| **Label Drift** | Changes target distribution | Market shifts, behavior changes |
| **Covariate Shift** | Changes feature distribution | New user demographics |
| **Concept Drift** | Changes feature→target relationship | Changing user preferences |

### Decision Tree: Which Drift Type?

```
Do you want to change...
├─ Feature values?
│  ├─ Gradually over time? → inject_feature_drift_gradual()
│  └─ Suddenly at a point? → inject_feature_drift_sudden()
├─ Target/label distribution?
│  └─ → inject_label_drift()
├─ Feature distributions (not values)?
│  └─ → inject_covariate_shift()
└─ Feature→Target relationship?
   └─ → inject_concept_drift()
```

### Basic Usage

```python
from calm_data_generator.injectors import DriftInjector

injector = DriftInjector()

# Gradual feature drift (most common)
drifted_data = injector.inject_feature_drift_gradual(
    data,
    feature_cols=["age", "income"],
    drift_type="shift",  # or "scale", "noise"
    drift_magnitude=0.3,
    start_index=100
)
```

---

## Drift Types Explained

### 1. Feature Drift (Gradual)

**What:** Feature values slowly change over time  
**Example:** Customer age increasing as your user base matures  
**Use Case:** Testing drift detection sensitivity

```python
# Simulate aging population
drifted = injector.inject_feature_drift_gradual(
    data,
    feature_cols=["age"],
    drift_type="shift",      # Shift mean upward
    drift_magnitude=0.5,     # 50% increase
    start_index=500,         # Start at row 500
    end_index=1000           # Complete by row 1000
)
```

**Drift Types:**
- `shift`: Changes mean (μ → μ + δ)
- `scale`: Changes variance (σ → σ × k)
- `noise`: Adds random noise

### 2. Feature Drift (Sudden)

**What:** Abrupt change at a specific point  
**Example:** New data collection system deployed  
**Use Case:** Testing alert systems

```python
# Simulate system update
drifted = injector.inject_feature_drift_sudden(
    data,
    feature_cols=["sensor_reading"],
    drift_type="shift",
    drift_magnitude=2.0,
    drift_point=750          # Change at row 750
)
```

### 3. Label Drift

**What:** Target variable distribution changes  
**Example:** Fraud rate increases from 1% to 5%  
**Use Case:** Testing model retraining triggers

```python
# Simulate fraud increase
drifted = injector.inject_label_drift(
    data,
    target_col="is_fraud",
    new_distribution={0: 0.95, 1: 0.05},  # 5% fraud
    start_index=600
)
```

### 4. Covariate Shift

**What:** Feature distributions change (not relationships)  
**Example:** New customer segment with different demographics  
**Use Case:** Testing domain adaptation

```python
# Simulate new user segment
drifted = injector.inject_covariate_shift(
    data,
    feature_cols=["age", "income"],
    shift_magnitude=1.5,
    start_index=400
)
```

### 5. Concept Drift

**What:** Feature→Target relationship changes  
**Example:** What makes a "good customer" changes  
**Use Case:** Testing model performance degradation

```python
# Simulate changing preferences
drifted = injector.inject_concept_drift(
    data,
    feature_cols=["price", "quality"],
    target_col="purchased",
    drift_magnitude=0.8,
    start_index=300
)
```

---

## Real-World Scenarios

### Scenario 1: E-Commerce Seasonal Drift

**Problem:** Customer behavior changes during holidays

```python
# Gradual increase in purchase amounts
drifted = injector.inject_feature_drift_gradual(
    sales_data,
    feature_cols=["purchase_amount"],
    drift_type="shift",
    drift_magnitude=0.4,  # 40% increase
    start_index=1000,     # Start of holiday season
    end_index=1500        # End of holiday season
)
```

### Scenario 2: Fraud Detection System Update

**Problem:** New fraud patterns emerge suddenly

```python
# Sudden change in fraud characteristics
drifted = injector.inject_feature_drift_sudden(
    transactions,
    feature_cols=["transaction_amount", "location_risk"],
    drift_type="shift",
    drift_magnitude=1.5,
    drift_point=2000
)
```

### Scenario 3: Credit Scoring Model Monitoring

**Problem:** Economic downturn changes default rates

```python
# Label drift: more defaults
drifted = injector.inject_label_drift(
    credit_data,
    target_col="default",
    new_distribution={0: 0.85, 1: 0.15},  # 15% default rate
    start_index=5000
)
```

---

A module to inject various types of drift (data shift) into datasets.

---

## Quick Start: Drift from `generate()`

You can inject drift directly when generating synthetic data using `RealGenerator.generate()`:

```python
from calm_data_generator.generators.tabular import RealGenerator

gen = RealGenerator()

synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    method='ctgan',
    drift_injection_config=[
        {
            "method": "inject_feature_drift_gradual",
            "params": {
                "feature_cols": ["age", "income"],
                "drift_type": "shift",
                "drift_magnitude": 0.3,
                "center": 500,
                "width": 200
            }
        },
        {
            "method": "inject_label_drift",
            "params": {
                "target_cols": ["label"],
                "drift_magnitude": 0.1
            }
        }
    ]
)
```

Each item in `drift_injection_config` requires:
- `method`: Name of the DriftInjector method (see below)
- `params`: Dictionary of parameters for that method

---

## DriftConfig Class Reference

**Import:** `from calm_data_generator.generators.configs import DriftConfig`

`DriftConfig` is a Pydantic model that provides type-safe configuration for drift injection. It supports both dictionary and object-based configuration.

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"inject_feature_drift"` | DriftInjector method to call |
| `drift_type` | str | `"gaussian_noise"` | Type of drift operation (see Operation Types) |
| `feature_cols` | List[str] | `None` | Columns to apply drift to |
| `magnitude` | float | `0.2` | Drift intensity (0.0-1.0 typical) |

### Selection Parameters (Row/Time Range)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_index` | int | `None` | Starting row index for drift |
| `end_index` | int | `None` | Ending row index for drift |
| `block_index` | int | `None` | Specific block to apply drift (for block generators) |
| `block_column` | str | `None` | Column name identifying blocks |
| `time_start` | str | `None` | Start timestamp (ISO format) |
| `time_end` | str | `None` | End timestamp (ISO format) |

### Gradual Drift Parameters (Window/Profile)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center` | int/float | `None` | Center point of drift window |
| `width` | int/float | `None` | Width of drift transition |
| `profile` | str | `"sigmoid"` | Transition shape: `"sigmoid"`, `"linear"`, `"cosine"` |
| `speed_k` | float | `1.0` | Transition speed multiplier |
| `direction` | str | `"up"` | Drift direction: `"up"` or `"down"` |
| `inconsistency` | float | `0.0` | Random noise in drift application (0.0-1.0) |

### Specialized Drift Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drift_value` | float | `None` | Fixed value for `add_value`, `multiply_value`, etc. |
| `drift_values` | Dict[str, float] | `None` | Per-column drift values |
| `params` | Dict[str, Any] | `{}` | Additional method-specific parameters |

### Usage Examples

**Basic Usage (Object):**
```python
from calm_data_generator.generators.configs import DriftConfig

drift_config = DriftConfig(
    method="inject_feature_drift",
    feature_cols=["age", "income"],
    drift_type="shift",
    magnitude=0.3,
    start_index=100,
    end_index=500
)
```

**Gradual Drift with Window:**
```python
drift_config = DriftConfig(
    method="inject_feature_drift_gradual",
    feature_cols=["temperature"],
    drift_type="shift",
    magnitude=0.5,
    center=500,      # Drift peaks at row 500
    width=200,       # Transition over 200 rows
    profile="sigmoid"  # Smooth S-curve transition
)
```

**Backward Compatibility (Dictionary):**
```python
# Still supported for backward compatibility
drift_config = {
    "method": "inject_feature_drift",
    "params": {
        "feature_cols": ["age"],
        "drift_magnitude": 0.3
    }
}
```

**Using with RealGenerator:**
```python
synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    drift_injection_config=[
        DriftConfig(
            method="inject_drift",
            params={
                "columns": ["age", "income"],
                "drift_mode": "gradual",
                "drift_magnitude": 0.3
            }
        )
    ]
)
```

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

## Unified Drift Injection: `inject_drift()`

**NEW!** A single method that auto-detects column types and applies appropriate drift operations.

```python
drifted = injector.inject_drift(
    df=data,
    columns=['age', 'income', 'gender', 'is_active'],  # Any column types
    drift_mode='gradual',          # 'abrupt', 'gradual', 'incremental', 'recurrent'
    drift_magnitude=0.3,
    center=500,                    # For gradual mode
    width=200,
)
```

### Auto-Detected Column Types

| Column Type | Detection | Default Operation |
|-------------|-----------|-------------------|
| **Numeric** | `int`, `float` dtypes | `shift` |
| **Categorical** | `object`, `category` dtypes | `frequency` |
| **Boolean** | `bool` dtype or 2 unique values | `flip` |

### Drift Modes

| Mode | Description |
|------|-------------|
| `abrupt` | Immediate change from `start_index` |
| `gradual` | Smooth transition using window function (sigmoid, linear, cosine) |
| `incremental` | Constant smooth drift over entire range |
| `recurrent` | Multiple drift windows (controlled by `repeats`) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | List[str] | - | Columns to apply drift (any type) |
| `drift_magnitude` | float | `0.3` | Intensity of drift (0.0 to 1.0) |
| `drift_mode` | str | `"abrupt"` | Type of drift pattern |
| `numeric_operation` | str | `"shift"` | Operation for numeric columns |
| `categorical_operation` | str | `"frequency"` | Operation for categorical columns |
| `boolean_operation` | str | `"flip"` | Operation for boolean columns |
| `center` | int | auto | Center of transition window (gradual) |
| `width` | int | auto | Width of transition window (gradual) |
| `profile` | str | `"sigmoid"` | Transition profile: `sigmoid`, `linear`, `cosine` |
| `repeats` | int | `3` | Number of windows (recurrent) |
| `start_index` | int | `None` | Row index where drift starts |
| `conditions` | List[Dict] | `None` | Conditional drift filters |

### Available Operations

**Numeric**: `shift`, `scale`, `gaussian_noise`, `uniform_noise`, `add_value`, `subtract_value`, `multiply_value`

**Categorical**: `frequency`, `new_category`, `typos`

**Boolean**: `flip`

### Examples

```python
# Abrupt drift on mixed columns
drifted = injector.inject_drift(
    df=data,
    columns=['age', 'income', 'category', 'is_active'],
    drift_mode='abrupt',
    drift_magnitude=0.3,
    start_index=500,
)

### Correlation Propagation

Drift injection can respect the correlation structure of your data, ensuring that changes in one feature are realistically reflected in correlated features.

```python
# Propagate drift to correlated features
drifted = injector.inject_drift(
    df=data,
    columns=['income'],
    drift_mode='gradual',
    drift_magnitude=0.2,
    correlations=True # Calculate and use current correlations
)
```

**`correlations` Parameter:**
- **`True`**: Calculates the Pearson correlation matrix from the current DataFrame and propagates drift to all correlated features proportionally.
- **`pd.DataFrame`** or **`Dict`**: Uses a distinct correlation matrix or dictionary you provide.
- **`None`** (Default): No propagation is performed; only the specified columns change.

Mechanism: $\Delta Y = \rho_{XY} \cdot \frac{\sigma_Y}{\sigma_X} \cdot \Delta X$

> [!TIP]
> **Note on Concept Drift:** If you want to simulate **Concept Drift** (where the model rules change and thus it fails, e.g., input grows but target stays low), you should **exclude the target column** from the correlation matrix or manually zero out its correlation before passing it. If you include the target in propagation, it will adjust along with the input (Covariate Shift), maintaining the original relationship and making the drift harder for the model to detect.

---

# Gradual drift with custom operations
drifted = injector.inject_drift(
    df=data,
    columns=['temperature', 'humidity', 'sensor_status'],
    drift_mode='gradual',
    drift_magnitude=0.5,
    numeric_operation='scale',
    boolean_operation='flip',
    center=1000,
    width=300,
)

# Recurrent drift (IoT sensor simulation)
drifted = injector.inject_drift(
    df=data,
    columns=['voltage', 'current'],
    drift_mode='recurrent',
    drift_magnitude=0.4,
    repeats=5,
)

# Use from generate() with drift_injection_config (List of DriftConfig objects)
from calm_data_generator.generators.configs import DriftConfig

synthetic = gen.generate(
    data=real_data,
    n_samples=1000,
    drift_injection_config=[
        DriftConfig(
            method="inject_drift",
            params={
                "columns": ["age", "income", "status"],
                "drift_mode": "gradual",
                "drift_magnitude": 0.3,
            }
        )
    ]
)

# Preventing Negative Values (e.g., Salary, Age)
# Use 'scale' (multiplication) instead of 'shift' (addition) to ensure values stay positive.
# Shift could subtract enough to make low values negative.
drifted = injector.inject_drift(
    df=data,
    columns=['salary', 'age'],
    drift_mode='gradual',
    drift_magnitude=0.2,       # Increases values by ~20%
    numeric_operation='scale'  # Safe for non-negative data
)
```

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

For **categorical** columns:

| Type | Description | Mechanism |
|------|-------------|-----------|
| `frequency` | Change frequency distribution | Resamples values favoring rare categories or inverting frequencies. |
| `new_category` | Introduce new value | Replaces values with a new category (e.g. `NEW_CAT`) with prob `magnitude`. |
| `typos` | Simulate typos | Introduces character-level noise (typos) into string values. |

For **boolean/label** columns:

| Type | Description | Mechanism |
|------|-------------|-----------|
| `flip` | Flip value | Inverts boolean value (`True` -> `False`, `1` -> `0`) with prob `magnitude`. |


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
