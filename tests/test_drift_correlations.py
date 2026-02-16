import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.drift import DriftInjector


@pytest.fixture
def correlated_data():
    """Create data with strong correlation: Y = 2*X + noise"""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + np.random.normal(0, 0.1, 100)  # Strong positive correlation
    z = np.random.normal(0, 1, 100)  # Uncorrelated
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def test_drift_propagation_positive(correlated_data):
    """Test that drift in X propagates to Y (correlated) but not Z (uncorrelated)."""
    injector = DriftInjector(auto_report=False)

    # Inject massive shift to X
    drift_magnitude = 5.0  # relative to mean/std depending on type
    # For 'shift', drift amount = magnitude * mean. mean(X) ~ 5. Shift ~ 25.

    # We expect Y to shift as well because corr(X,Y) ~ 1.
    # Delta_Y ~ rho * (std_Y / std_X) * Delta_X
    # rho ~ 1
    # std_Y ~ 2 * std_X
    # Delta_Y ~ 1 * 2 * Delta_X = 2 * Delta_X

    result = injector.inject_feature_drift(
        df=correlated_data.copy(),
        feature_cols=["X"],
        drift_type="shift",
        drift_magnitude=drift_magnitude,
        correlations=True,
    )

    # Check X drift
    x_orig = correlated_data["X"]
    x_new = result["X"]
    delta_x = (x_new - x_orig).mean()
    assert abs(delta_x) > 1.0  # Verify significant drift occurred

    # Check Y drift (Should be ~ 2 * delta_x)
    y_orig = correlated_data["Y"]
    y_new = result["Y"]
    delta_y = (y_new - y_orig).mean()

    # Allow some margin due to noise and floating point
    expected_delta_y = 2.0 * delta_x
    assert np.isclose(delta_y, expected_delta_y, rtol=0.1)

    # Check Z drift (Should be ~ 0)
    z_orig = correlated_data["Z"]
    z_new = result["Z"]
    delta_z = (z_new - z_orig).mean()
    assert abs(delta_z) < 0.1  # Should be negligible


def test_drift_propagation_disabled(correlated_data):
    """Test standard drift without correlation propagation."""
    injector = DriftInjector(auto_report=False)

    result = injector.inject_feature_drift(
        df=correlated_data.copy(),
        feature_cols=["X"],
        drift_type="shift",
        drift_magnitude=5.0,
        correlations=False,
    )

    # Y should NOT change
    assert np.allclose(result["Y"], correlated_data["Y"])
