import pandas as pd
import numpy as np
import pytest
from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector


class TestCorrelationPropagation:
    @pytest.fixture
    def correlated_data(self):
        np.random.seed(42)
        n = 1000
        x = np.random.normal(0, 1, n)
        # Y = 2X + noise. Correlation should be high (~0.99)
        y = 2 * x + np.random.normal(0, 0.01, n)
        # Z is uncorrelated
        z = np.random.normal(0, 1, n)
        return pd.DataFrame({"X": x, "Y": y, "Z": z})

    def test_drift_injector_propagation(self, correlated_data):
        df = correlated_data.copy()
        injector = DriftInjector(auto_report=False)
        corr_matrix = df.corr()

        # Inject exact additive shift in X with propagation
        # Using 'add_value' instead of 'shift' because 'shift' depends on mean, which is ~0 here.
        drift_magnitude = (
            1.0  # Not used for add_value logic usually, but we pass drift_value
        )

        df_drifted = injector.inject_feature_drift(
            df,
            ["X"],
            drift_type="add_value",
            drift_value=1.0,
            correlations=corr_matrix,
        )

        delta_x = df_drifted["X"].mean() - df["X"].mean()
        delta_y = df_drifted["Y"].mean() - df["Y"].mean()
        delta_z = df_drifted["Z"].mean() - df["Z"].mean()

        # Check X shifted by exactly 1.0
        assert np.isclose(delta_x, 1.0, atol=0.1)

        # Check Y shifted proportionally
        # Expected Delta Y = Rho * (StdY/StdX) * Delta X
        # Rho ~ 1, StdY ~ 2, StdX ~ 1 => Factor ~ 2
        expected_delta_y = (
            corr_matrix.loc["X", "Y"] * (df["Y"].std() / df["X"].std()) * delta_x
        )
        assert np.isclose(delta_y, expected_delta_y, atol=0.1)
        assert delta_y > 1.5  # Should be significant

        # Check Z did NOT shift significantly (uncorrelated)
        assert np.abs(delta_z) < 0.1

    def test_drift_injector_gradual_propagation(self, correlated_data):
        df = correlated_data.copy()
        injector = DriftInjector(auto_report=False)
        corr_matrix = df.corr()

        # Inject gradual drift in X. Using add_value for consistency.
        df_drifted = injector.inject_feature_drift_gradual(
            df,
            ["X"],
            drift_type="add_value",
            drift_value=1.0,
            correlations=corr_matrix,
        )

        # The mean shift will be less than 1.0 because it's gradual, but property should hold
        # Wait, gradual drift with add_value might not be supported or behavior is weighted add.
        # w * drift_value.
        delta_x = df_drifted["X"].mean() - df["X"].mean()
        delta_y = df_drifted["Y"].mean() - df["Y"].mean()

        expected_delta_y = (
            corr_matrix.loc["X", "Y"] * (df["Y"].std() / df["X"].std()) * delta_x
        )
        assert np.isclose(delta_y, expected_delta_y, atol=0.1)

    def test_scenario_injector_propagation(self, correlated_data):
        df = correlated_data.copy()
        scenario = ScenarioInjector(minimal_report=True)
        corr_matrix = df.corr()

        evolution_config = {"X": {"type": "linear", "slope": 0.01}}

        df_evolved = scenario.evolve_features(
            df,
            evolution_config,
            correlations=corr_matrix,
            auto_report=False,
        )

        delta_x = df_evolved["X"].mean() - df["X"].mean()
        delta_y = df_evolved["Y"].mean() - df["Y"].mean()

        expected_delta_y = (
            corr_matrix.loc["X", "Y"] * (df["Y"].std() / df["X"].std()) * delta_x
        )
        assert np.isclose(delta_y, expected_delta_y, atol=0.1)
        assert delta_y > 0.1  # Should have moved

    def test_propagation_disabled(self, correlated_data):
        df = correlated_data.copy()
        injector = DriftInjector(auto_report=False)

        df_drifted = injector.inject_feature_drift(
            df,
            ["X"],
            drift_type="add_value",
            drift_value=1.0,
        )

        delta_y = df_drifted["Y"].mean() - df["Y"].mean()
        assert np.abs(delta_y) < 0.1  # Should not change significantly
