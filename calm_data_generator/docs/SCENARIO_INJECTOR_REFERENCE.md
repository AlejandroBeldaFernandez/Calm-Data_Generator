# ScenarioInjector - Complete Reference

**Location:** `calm_data_generator.injectors.ScenarioInjector`

Tool for injecting dynamic scenarios and temporal patterns into datasets.

---

## Quick Start Guide

### What is ScenarioInjector?

A tool to simulate **temporal dynamics** and **evolving patterns** in data. Different from DriftInjector:
- **DriftInjector**: Changes distributions (what the data looks like)
- **ScenarioInjector**: Creates patterns/scenarios (how features evolve)

### When to Use ScenarioInjector vs DriftInjector

| Use ScenarioInjector When | Use DriftInjector When |
|---------------------------|------------------------|
| Creating **feature evolution** patterns | Simulating **distribution changes** |
| Building **temporal dependencies** | Testing **drift detection** |
| Simulating **scenarios** (growth, decay) | Monitoring **model performance** |
| Creating **target from features** | Changing **existing distributions** |

### Decision Tree: Which Tool?

```
What do you want to do?
├─ Create NEW patterns/scenarios?
│  ├─ Feature evolution over time? → ScenarioInjector.inject_feature_evolution()
│  ├─ Construct target from features? → ScenarioInjector.construct_target_from_features()
│  └─ Add temporal dynamics? → ScenarioInjector
└─ Change EXISTING distributions?
   ├─ Test drift detection? → DriftInjector
   ├─ Simulate data quality issues? → DriftInjector
   └─ Model monitoring? → DriftInjector
```

### Basic Usage

```python
from calm_data_generator.injectors import ScenarioInjector

injector = ScenarioInjector()

# Create growth pattern
evolved_data = injector.inject_feature_evolution(
    data,
    feature_col="revenue",
    evolution_type="exponential_growth",
    growth_rate=0.05  # 5% growth
)
```

---

## Scenario Types Explained

### 1. Feature Evolution

**What:** Creates temporal patterns in features  
**Use Cases:** Revenue growth, user engagement decay, seasonal patterns

#### Available Evolution Types:

| Type | Pattern | Use Case |
|------|---------|----------|
| `linear_growth` | Steady increase | Sales growth, user acquisition |
| `exponential_growth` | Accelerating increase | Viral growth, compound interest |
| `logarithmic_growth` | Slowing increase | Market saturation, learning curves |
| `decay` | Decrease over time | Churn, engagement drop |
| `seasonal` | Cyclical pattern | Retail seasons, weather |
| `step_function` | Discrete jumps | Product launches, policy changes |

#### Examples:

**Exponential Growth (Startup Metrics)**
```python
# Simulate viral user growth
evolved = injector.inject_feature_evolution(
    data,
    feature_col="daily_active_users",
    evolution_type="exponential_growth",
    growth_rate=0.1,  # 10% daily growth
    start_index=0
)
```

**Seasonal Pattern (Retail Sales)**
```python
# Simulate holiday shopping patterns
evolved = injector.inject_feature_evolution(
    data,
    feature_col="sales",
    evolution_type="seasonal",
    period=365,  # Yearly cycle
    amplitude=0.3  # 30% variation
)
```

**Decay (User Engagement)**
```python
# Simulate engagement drop after feature removal
evolved = injector.inject_feature_evolution(
    data,
    feature_col="session_duration",
    evolution_type="decay",
    decay_rate=0.05,  # 5% decay per period
    start_index=500
)
```

### 2. Target Construction

**What:** Creates target variable from feature combinations  
**Use Cases:** Synthetic labels, complex decision rules, multi-factor outcomes

```python
# Construct "high_value_customer" from multiple features
data_with_target = injector.construct_target_from_features(
    data,
    target_col="high_value",
    feature_weights={
        "purchase_frequency": 0.4,
        "avg_order_value": 0.3,
        "customer_lifetime": 0.3
    },
    threshold=0.7  # Top 30% are "high value"
)
```

---

## Real-World Scenarios

### Scenario 1: SaaS Company Growth

**Problem:** Simulate realistic startup growth metrics

```python
# Monthly recurring revenue (MRR) with exponential growth
data = injector.inject_feature_evolution(
    data,
    feature_col="mrr",
    evolution_type="exponential_growth",
    growth_rate=0.15,  # 15% monthly growth
    noise_level=0.05   # 5% random variation
)

# Churn rate decreasing as product matures
data = injector.inject_feature_evolution(
    data,
    feature_col="churn_rate",
    evolution_type="logarithmic_decay",
    decay_rate=0.1
)
```

### Scenario 2: E-Commerce Seasonality

**Problem:** Model holiday shopping patterns

```python
# Sales with strong seasonal component
data = injector.inject_feature_evolution(
    data,
    feature_col="daily_sales",
    evolution_type="seasonal",
    period=365,        # Yearly cycle
    amplitude=0.5,     # 50% variation
    peaks=[335, 350]   # Black Friday, Christmas
)
```

### Scenario 3: Product Launch Impact

**Problem:** Simulate step-change from new feature

```python
# User engagement jumps after feature launch
data = injector.inject_feature_evolution(
    data,
    evolution_type="step_function",
    feature_col="engagement_score",
    step_points=[1000],  # Launch at row 1000
    step_values=[1.5]    # 50% increase
)
```

### Scenario 4: Credit Risk Modeling

**Problem:** Create credit score from multiple factors

```python
# Construct credit risk from financial indicators
data = injector.construct_target_from_features(
    data,
    target_col="credit_risk",
    feature_weights={
        "income": 0.3,
        "debt_to_income": -0.4,  # Negative weight
        "payment_history": 0.3
    },
    threshold=0.6,  # Above 0.6 = "low risk"
    binary=True
)
```

---

## Combining with DriftInjector

You can use both tools together for complex scenarios:

```python
from calm_data_generator.injectors import ScenarioInjector, DriftInjector

scenario = ScenarioInjector()
drift = DriftInjector()

# 1. Create growth pattern
data = scenario.inject_feature_evolution(
    data,
    feature_col="users",
    evolution_type="exponential_growth",
    growth_rate=0.1
)

# 2. Add sudden drift (system change)
data = drift.inject_feature_drift_sudden(
    data,
    feature_cols=["users"],
    drift_type="shift",
    drift_magnitude=0.5,
    drift_point=500
)
```

---

## Industry-Specific Examples

### Healthcare: Disease Progression
```python
# Simulate biomarker decay in treatment study
data = injector.inject_feature_evolution(
    patient_data,
    feature_col="tumor_marker",
    evolution_type="decay",
    decay_rate=0.08,  # 8% reduction per month
    start_index=0  # Treatment starts immediately
)
```

### Finance: Market Trends
```python
# Simulate stock price with trend + seasonality
data = injector.inject_feature_evolution(
    stock_data,
    feature_col="price",
    evolution_type="linear_growth",
    growth_rate=0.02  # 2% monthly growth
)
```

### IoT: Sensor Degradation
```python
# Simulate sensor accuracy decay
data = injector.inject_feature_evolution(
    sensor_data,
    feature_col="accuracy",
    evolution_type="logarithmic_decay",
    decay_rate=0.05
)
```

---

## ScenarioConfig Class Reference

**Import:** `from calm_data_generator.generators.configs import ScenarioConfig, EvolutionFeatureConfig`

`ScenarioConfig` is a Pydantic model for configuring scenario injection with feature evolution and target construction.

### ScenarioConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_config` | Dict | `None` | State configuration for scenario |
| `evolve_features` | Dict[str, Union[Dict, EvolutionFeatureConfig]] | `{}` | Feature evolution configurations |
| `construct_target` | Dict | `None` | Target construction configuration |

### EvolutionFeatureConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Evolution type: `"linear"`, `"cycle"`, `"sigmoid"`, `"trend"`, `"seasonal"`, `"noise"`, `"decay"` |
| `slope` | float | `0.0` | Slope for linear/trend evolution |
| `intercept` | float | `0.0` | Intercept for linear evolution |
| `amplitude` | float | `1.0` | Amplitude for seasonal/cycle patterns |
| `period` | float | `100.0` | Period length for cyclical patterns |
| `phase` | float | `0.0` | Phase shift for cyclical patterns |
| `center` | float | `None` | Center point for sigmoid evolution |
| `width` | float | `None` | Width for sigmoid transition |

### Usage Examples

**Basic Feature Evolution (Object):**
```python
from calm_data_generator.generators.configs import EvolutionFeatureConfig

evolution_config = {
    "revenue": EvolutionFeatureConfig(
        type="linear",
        slope=100.0,  # Increase by 100 per period
        intercept=1000.0
    ),
    "temperature": EvolutionFeatureConfig(
        type="seasonal",
        amplitude=10.0,
        period=365,  # Yearly cycle
        phase=0.0
    )
}
```

**Using ScenarioConfig:**
```python
from calm_data_generator.generators.configs import ScenarioConfig, EvolutionFeatureConfig

scenario_config = ScenarioConfig(
    evolve_features={
        "sales": EvolutionFeatureConfig(
            type="trend",
            slope=0.05  # 5% growth
        ),
        "churn": EvolutionFeatureConfig(
            type="decay",
            slope=-0.02  # 2% decay
        )
    },
    construct_target={
        "formula": "0.3 * sales - 0.5 * churn",
        "threshold": 0.7
    }
)
```

**Backward Compatibility (Dictionary):**
```python
# Still supported
evolution_config = {
    "price": {
        "type": "trend",
        "slope": 0.01
    }
}
```

---

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
