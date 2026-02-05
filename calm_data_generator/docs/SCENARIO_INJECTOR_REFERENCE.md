# ScenarioInjector - Complete Reference

**Location:** `calm_data_generator.generators.dynamics.ScenarioInjector`

A module to evolve features, build targets based on rules, and project data to future time periods.

---

## Initialization

```python
from calm_data_generator.generators.dynamics import ScenarioInjector

scenario = ScenarioInjector(
    seed=42,                    # Seed for reproducibility
    minimal_report=False,       # Full reports
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | `None` | Seed for reproducibility |
| `minimal_report` | bool | `False` | If True, simplified reports |

---

## Method: `evolve_features()`

Evolves numeric columns based on configurations like trends, seasonality, noise, etc.

### Basic Syntax

```python
evolved_df = scenario.evolve_features(
    df=df,                                    # Input DataFrame
    evolution_config={...},                   # Evolution config per column
    time_col="date",                          # Time column (optional)
    output_dir="./output",                    # Output directory
    auto_report=True,                         # Generate report
    generator_name="ScenarioInjector",        # File base name
    auto_report=True,                         # Generate report
    generator_name="ScenarioInjector",        # File base name
    resample_rule=None,                       # Time resampling rule
    correlations=None,                        # Drift propagation control
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | - | Original dataset (required) |
| `evolution_config` | Dict | - | Evolution configuration per column |
| `time_col` | str | `None` | Time column for sorting |
| `output_dir` | str | `None` | Output directory |
| `auto_report` | bool | `True` | Automatically generate a report |
| `generator_name` | str | `"ScenarioInjector"` | Base name for output files |
| `resample_rule` | str/int | `None` | Resampling rule (e.g., "D", "W") |
| `correlations` | bool/df/dict | `None` | If `True` or a matrix, propagates evolution to correlated features. |

### Evolution Types

#### 1. Linear Trend (`trend`)

Adds a growing or decreasing linear slope:

```python
evolution_config = {
    "price": {
        "type": "trend",
        "slope": 0.01,      # Increment per row (positive = growth)
    }
}
```

**Parameters:**
- `slope` (float): Tendency slope. Positive = increasing, Negative = decreasing.

**Formula:** `new_value = old_value + (slope * row_index)`

#### 2. Seasonality (`seasonal`)

Adds a periodic sinusoidal pattern:

```python
evolution_config = {
    "temperature": {
        "type": "seasonal",
        "amplitude": 5.0,     # Maximum height from center
        "period": 365,        # Rows per full cycle
        "phase": 0,           # Optional phase shift
    }
}
```

**Formula:** `new_value = old_value + amplitude * sin(2π * row_index / period + phase)`

#### 3. Gaussian Noise (`noise`)

Adds random additive noise:

```python
evolution_config = {
    "reading": {
        "type": "noise",
        "scale": 0.1,         # Noise standard deviation
    }
}
```

**Formula:** `new_value = old_value + N(0, scale)`

#### 4. Exponential Decay (`decay`)

Applies an exponential decay factor:

```python
evolution_config = {
    "battery": {
        "type": "decay",
        "rate": 0.01,         # Decay rate per row
    }
}
```

**Formula:** `new_value = old_value * (1 - rate) ^ row_index`

---

## Method: `construct_target()`

Creates or overwrites a target variable based on user-defined formulas.

### Basic Syntax

```python
df_with_target = scenario.construct_target(
    df=df,                              # Input DataFrame
    target_col="risk_score",            # Target column name
    formula="...",                      # String or callable formula
    noise_std=0.0,                      # Additive Gaussian noise
    task_type="regression",             # "regression" or "classification"
    threshold=None,                     # Threshold for binary output
)
```

### Formula Types

#### 1. String Formula

Mathematical expression referencing existing columns:

```python
df = scenario.construct_target(
    df=df,
    target_col="risk_score",
    formula="0.3 * age + 0.5 * bmi - 0.2 * exercise_hours",
)
```

#### 2. Callable Formula

A function that takes a row and returns a value:

```python
def calculate_risk(row):
    return row["age"] * 0.01 * (2 if row["smoker"] == 1 else 1)

df = scenario.construct_target(
    df=df,
    target_col="risk_score",
    formula=calculate_risk,
)
```

---

## Method: `project_to_future_period()`

Projects historical data into future time periods by generating synthetic data and applying trends.

### Basic Syntax

```python
future_df = scenario.project_to_future_period(
    df=df,                              # Historical DataFrame
    periods=12,                         # Number of future periods
    time_col="month",                   # Time column
    evolution_config={...},             # Trends to apply
    generator_method="ctgan",           # Method for synthetic base
    n_samples_per_period=100,           # Samples per future period
)
```

### Internal Workflow

1. **Synthetic base generation** using `RealGenerator`.
2. **Future time period assignment** sequentially.
3. **Application of `evolve_features()`** for requested trends.
4. **Report generation** comparing historical vs. projected data.

---

## Use Cases

### 1. What-If Scenario Simulation

```python
# Scenario: What if prices increase by 10% and satisfaction decays?
scenario_df = scenario.evolve_features(
    df=baseline_df,
    evolution_config={
        "price": {"type": "trend", "slope": 0.001},
        "satisfaction": {"type": "decay", "rate": 0.002},
    },
)

# Recalculate target
scenario_df = scenario.construct_target(
    df=scenario_df,
    target_col="churn_prob",
    formula="0.3 * (1 - satisfaction) + 0.2 * price",
)
```

### 2. Known-Relationship Dataset Generation

```python
# Target with known linear dependency
df = scenario.construct_target(
    df=df,
    target_col="y",
    formula="2 * x1 + 0.5 * x2 - 1.5 * x3",
    noise_std=0.1,
)
```

---

## Comprehensive Use Cases

### Case 1: Economic Recession Impact

**Scenario:** Simulate a market downturn where purchasing power decreases and default risk rises.

**Solution:** Apply negative trend to income and positive trend to risk.

```python
from calm_data_generator.generators.dynamics import ScenarioInjector

scenario = ScenarioInjector()

# Evolve features to simulate recession over time
recession_df = scenario.evolve_features(
    df=economic_df,
    evolution_config={
        "avg_income": {"type": "trend", "slope": -50.0},  # Income drops $50 per day
        "unemployment_rate": {"type": "trend", "slope": 0.01}, # Unemployment rises
        "market_index": {"type": "decay", "rate": 0.005}  # Market crashes exponentially
    },
    time_col="date"
)

# Re-calculate default probability based on new economic conditions
recession_df = scenario.construct_target(
    df=recession_df,
    target_col="default_prob",
    # Rule: Low income & high unemployment = high default risk
    formula="0.7 * (1/avg_income) + 0.5 * unemployment_rate",
    task_type="regression"
)
```

### Case 2: Seasonal Sales Projection

**Scenario:** A retailer wants to project sales for the next year based on historical data, accounting for the Christmas spike.

**Solution:** Use `project_to_future_period` with seasonality.

```python
future_sales = scenario.project_to_future_period(
    df=historical_sales_df,
    periods=12,              # Project 12 future months
    time_col="month",
    n_samples_per_period=5000,
    evolution_config={
        "revenue": {
            "type": "seasonal",
            "amplitude": 50000, # Large spike
            "period": 12,       # Yearly cycle (12 months)
            "phase": 3          # Shift peak to occur in December
        },
        "customer_count": {
            "type": "trend",
            "slope": 100        # Steady growth of 100 customers/month
        }
    }
)
```

### Case 3: Behavioral Change (Policy Shift)

**Scenario:** An insurance company changes its policy logic. The "Risk Score" definition changes completely from one day to another.

**Solution:** Construct a new target with complex logic.

```python
def new_policy_logic(row):
    # New logic: Penalize high speed more than before, ignore age
    base_score = row["avg_speed"] * 0.5
    if row["accidents_history"] > 0:
        base_score *= 2.0
    return min(base_score, 100)

new_policy_df = scenario.construct_target(
    df=drivers_df,
    target_col="risk_score_v2",
    formula=new_policy_logic,
    noise_std=2.0  # Add some variability
)
```
