import pandas as pd
import numpy as np
import tempfile
import os
import pytest


@pytest.fixture
def sample_data():
    """Create sample data with more balanced classes for SMOTE/ADASYN."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(20, 70, 100),
            "income": np.random.normal(50000, 15000, 100).astype(int),
            "score": np.random.uniform(0, 100, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100, p=[0.6, 0.4]),
        }
    )


def test_real_generator_methods(sample_data):
    """Test RealGenerator with multiple synthesis methods."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    # CART
    synth = gen.generate(sample_data, 20, method="cart", target_col="target")
    assert synth is not None and len(synth) == 20

    # RF
    synth = gen.generate(sample_data, 20, method="rf", target_col="target")
    assert synth is not None and len(synth) == 20

    # LGBM (low epochs)
    synth = gen.generate(
        sample_data, 20, method="lgbm", target_col="target", n_estimators=5
    )
    assert synth is not None and len(synth) == 20

    # Resample
    synth = gen.generate(sample_data, 20, method="resample", target_col="target")
    assert synth is not None and len(synth) == 20

    # SMOTE
    numeric_data = sample_data[["age", "income", "score", "target"]].copy()
    synth = gen.generate(numeric_data, 30, method="smote", target_col="target")
    assert synth is not None and len(synth) == 30


def test_clinical_data_generator():
    """Test basic ClinicalDataGenerator functionality."""
    from calm_data_generator.generators.clinical import ClinicalDataGenerator

    clin_gen = ClinicalDataGenerator()
    result = clin_gen.generate(n_samples=10, n_genes=20, n_proteins=10)
    assert "demographics" in result
    assert len(result["demographics"]) == 10


def test_drift_injector(sample_data):
    """Test standard drift injection methods."""
    from calm_data_generator.generators.drift import DriftInjector

    injector = DriftInjector()
    # Gradual drift
    drifted = injector.inject_feature_drift_gradual(
        df=sample_data.copy(),
        feature_cols=["score"],
        drift_magnitude=0.5,
        drift_type="shift",
        start_index=50,
        center=25,
        width=20,
    )
    assert len(drifted) == len(sample_data)

    # Feature drift
    drifted = injector.inject_feature_drift(
        df=sample_data.copy(),
        feature_cols=["income"],
        drift_magnitude=0.3,
        drift_type="shift",
        start_index=60,
    )
    assert len(drifted) == len(sample_data)


def test_scenario_injector(sample_data):
    """Test ScenarioInjector features."""
    from calm_data_generator.generators.dynamics import ScenarioInjector

    scenario = ScenarioInjector(seed=42)
    ts_data = sample_data.copy()
    ts_data["timestamp"] = pd.date_range("2024-01-01", periods=len(ts_data), freq="D")

    evolved = scenario.evolve_features(
        df=ts_data,
        evolution_config={"score": {"type": "trend", "rate": 0.05}},
        time_col="timestamp",
    )
    assert len(evolved) == len(ts_data)

    result = scenario.construct_target(
        df=sample_data.copy(),
        target_col="new_target",
        formula="age + income / 10000",
        task_type="regression",
    )
    assert "new_target" in result.columns


def test_anonymizer(sample_data):
    """Test data anonymization functions."""
    from calm_data_generator.anonymizer import (
        pseudonymize_columns,
        add_laplace_noise,
        generalize_numeric_to_ranges,
        shuffle_columns,
    )

    priv_data = sample_data.copy()
    priv_data["id"] = [f"P{i}" for i in range(len(priv_data))]
    result = pseudonymize_columns(priv_data, columns=["id"])
    assert "id" in result.columns

    result = add_laplace_noise(sample_data.copy(), columns=["age"], epsilon=1.0)
    assert len(result) == len(sample_data)

    result = generalize_numeric_to_ranges(
        sample_data.copy(),
        columns=["age"],
        num_bins=5,
    )
    assert len(result) == len(sample_data)


def test_single_call_workflow(sample_data):
    """Test Generate + Drift + Report in one call."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = gen.generate(
            data=sample_data,
            n_samples=20,
            method="cart",
            target_col="target",
            output_dir=tmpdir,
            save_dataset=True,
            drift_injection_config=[
                {
                    "method": "inject_feature_drift_gradual",
                    "params": {
                        "feature_cols": ["score"],
                        "drift_type": "shift",
                        "drift_magnitude": 0.3,
                        "start_index": 10,
                    },
                }
            ],
        )
        assert result is not None
        assert len(result) == 20
        assert len(os.listdir(tmpdir)) > 0
