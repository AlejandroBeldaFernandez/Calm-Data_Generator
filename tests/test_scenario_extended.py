import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.dynamics import ScenarioInjector


@pytest.fixture
def ts_data():
    """Create a time series dataframe."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "feature_a": np.random.normal(100, 10, 100),
            "feature_b": np.random.uniform(0, 1, 100),
            "category": np.random.choice(["X", "Y"], 100),
        }
    )
    return df


def test_feature_evolution_complex(ts_data):
    """Test multiple evolution types simultaneously."""
    injector = ScenarioInjector()

    # Apply linear trend to feature_a and sinusoidal to feature_b
    evolution_config = {
        "feature_a": {
            "type": "linear",
            "slope": 0.1,  # Increases by 0.1 per step (index)
        },
        "feature_b": {"type": "sinusoidal", "period": 20, "amplitude": 5.0},
    }

    # Pass time_col=None to use index steps (0, 1, 2...) for predictable evolution
    result = injector.evolve_features(
        df=ts_data, evolution_config=evolution_config, time_col=None
    )

    # Check trend
    # Original mean ~100. After 100 steps of 1% compound growth? Or linear?
    # Logic usually applies factor based on time index.
    # Let's verify it changed significantly and in positive direction
    assert result["feature_a"].mean() > ts_data["feature_a"].mean()

    # Check seasonality
    # Hard to check statistically on small sample without decomposition,
    # but variance should increase due to added wave?
    assert result["feature_b"].var() > ts_data["feature_b"].var()


def test_target_construction_formula(ts_data):
    """Test constructing target with complex formula."""
    injector = ScenarioInjector()

    # Valid formula
    # New target = feature_a * feature_b + constant
    formula = "feature_a * feature_b + 50"

    result = injector.construct_target(
        df=ts_data, target_col="new_kpi", formula=formula, task_type="regression"
    )

    assert "new_kpi" in result.columns

    # Verify calculation manually
    expected = ts_data["feature_a"] * ts_data["feature_b"] + 50
    # Note: ScenarioInjector might apply noise? Let's check exact match or close
    assert np.allclose(result["new_kpi"], expected, rtol=0.01)


def test_target_construction_conditional(ts_data):
    """Test conditional target logic."""
    injector = ScenarioInjector()

    # Create binary target based on threshold
    # if feature_a > 100 -> 1 else 0

    # Formula supports numpy/pandas expressions usually or numexpr
    # Let's try simple logic supported by implementation
    # If implementation uses pd.eval, we can use conditions

    formula = "(feature_a > 100) * 1"

    result = injector.construct_target(
        df=ts_data, target_col="churn_flag", formula=formula, task_type="classification"
    )

    assert "churn_flag" in result.columns
    assert set(result["churn_flag"].unique()).issubset({0, 1})

    # Check logic
    mask = ts_data["feature_a"] > 100
    assert np.all(result.loc[mask, "churn_flag"] == 1)
    assert np.all(result.loc[~mask, "churn_flag"] == 0)
