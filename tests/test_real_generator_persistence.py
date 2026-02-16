import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import joblib
from calm_data_generator.generators.tabular import RealGenerator


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randint(0, 10, 50),
            "target": np.random.choice([0, 1], 50),
        }
    )


def test_save_load_cart(sample_data):
    """Test persistence for CART method (simple sklearn model)."""
    gen = RealGenerator(auto_report=False)
    # Fit model implicitly by generating
    gen.generate(sample_data, n_samples=10, method="cart", target_col="target")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "cart_model.pkl")
        gen.save(save_path)

        assert os.path.exists(save_path)

        # Load back
        loaded_gen = RealGenerator.load(save_path)
        assert loaded_gen.method.lower() == "cart"

        # Generate from loaded
        new_samples = loaded_gen._generate_from_fitted(n_samples=5)
        assert len(new_samples) == 5
        assert set(new_samples.columns) == set(sample_data.columns)


def test_save_load_rf(sample_data):
    """Test persistence for Random Forest."""
    gen = RealGenerator(auto_report=False)
    gen.generate(sample_data, n_samples=10, method="rf", target_col="target")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(
            tmpdir, "rf_model.zip"
        )  # Use zip extension to test zip format
        gen.save(save_path)

        assert os.path.exists(save_path)

        loaded_gen = RealGenerator.load(save_path)
        assert loaded_gen.method.lower() == "rf"

        new_samples = loaded_gen._generate_from_fitted(n_samples=5)
        assert len(new_samples) == 5


def test_save_load_copula(sample_data):
    """Test persistence for Copula (requires specific handling in save/load)."""
    try:
        import copulae
    except ImportError:
        pytest.skip("copulae not installed")

    gen = RealGenerator(auto_report=False)
    # Copula only reliable on numeric data mostly
    numeric_data = sample_data.drop(columns=["target"])
    gen.generate(numeric_data, n_samples=10, method="copula")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "copula_model.pkl")
        gen.save(save_path)

        loaded_gen = RealGenerator.load(save_path)
        assert loaded_gen.method == "copula"

        # Check if loaded generator has metadata
        assert loaded_gen.metadata is not None
        assert "scaler" in loaded_gen.metadata

        new_samples = loaded_gen._generate_from_fitted(n_samples=5)
        assert len(new_samples) == 5


def test_load_nonexistent_file():
    """Test error handling for bad path."""
    with pytest.raises(FileNotFoundError):
        RealGenerator.load("non_existent_file_path_12345.pkl")
