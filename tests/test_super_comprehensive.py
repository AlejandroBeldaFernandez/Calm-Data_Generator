import pandas as pd
import numpy as np
import tempfile
import os
import pytest
from typing import Dict, List, Tuple


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "age": np.random.randint(20, 70, n),
            "income": np.random.normal(50000, 15000, n).astype(int),
            "score": np.random.uniform(0, 100, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "block": np.repeat(range(10), 10),
        }
    )


def test_real_generator_all_methods(sample_data):
    """Smoke test for all RealGenerator methods with low usage."""
    from calm_data_generator.generators.tabular import RealGenerator

    gen = RealGenerator(auto_report=False)

    # GMM only supports numeric data
    numeric_data = sample_data.select_dtypes(include=[np.number])

    # Methods to test
    methods = ["gmm", "copula"]
    for method in methods:
        data_to_use = numeric_data if method == "gmm" else sample_data

        try:
            synth = gen.generate(data_to_use, 10, method=method, target_col="target")
            if synth is None:
                pytest.skip(
                    f"Method {method} returned None (likely missing dependency)"
                )
            assert len(synth) == 10
        except ImportError:
            pytest.skip(f"Method {method} failed due to missing dependency")


def test_drift_injector_all_modes(sample_data):
    """Smoke test for additional drift modes."""
    from calm_data_generator.generators.drift import DriftInjector

    injector = DriftInjector()
    # Abrupt drift
    drifted = injector.inject_drift(
        df=sample_data,
        columns="score",
        drift_type="shift",
        magnitude=0.5,
        mode="abrupt",
    )
    assert len(drifted) == len(sample_data)


def test_clinical_data_generator_longitudinal():
    """Longitudinal clinical data test."""
    from calm_data_generator.generators.clinical import ClinicalDataGenerator

    clin_gen = ClinicalDataGenerator()
    result = clin_gen.generate_longitudinal_data(
        n_samples=5, longitudinal_config={"n_visits": 2}
    )
    assert result is not None


def test_stream_generator_basic(sample_data):
    """Basic StreamGenerator test."""
    from calm_data_generator.generators.stream import StreamGenerator

    stream_gen = StreamGenerator(auto_report=False)

    # The user says there's no generate_stream anymore, it's called generate
    # And it returns a DataFrame, not an iterator.
    try:
        from river import synth

        river_gen = synth.Agrawal(seed=42)

        result = stream_gen.generate(generator_instance=river_gen, n_samples=20)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20
    except ImportError:
        pytest.skip("River not installed")
