"""
Tests for scVI and scGen single-cell synthesis methods in RealGenerator.
"""

import pytest
import pandas as pd
import numpy as np
import importlib.util

scgen_available = importlib.util.find_spec("scgen") is not None


class TestSingleCellSynthesis:
    """Tests for scVI and scGen single-cell data generation."""

    @pytest.fixture
    def sample_expression_data(self):
        """Creates sample gene expression-like data."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        # Simulate count data (non-negative integers)
        expression = np.random.poisson(lam=5, size=(n_cells, n_genes))

        # Create DataFrame with gene names as columns
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        df = pd.DataFrame(expression, columns=gene_names)

        # Add cell type as target column
        df["cell_type"] = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_cells)

        return df

    @pytest.fixture
    def sample_expression_with_condition(self):
        """Creates sample gene expression data with condition labels."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        expression = np.random.poisson(lam=5, size=(n_cells, n_genes))
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        df = pd.DataFrame(expression, columns=gene_names)

        df["cell_type"] = np.random.choice(["TypeA", "TypeB"], size=n_cells)
        df["condition"] = np.random.choice(["control", "treated"], size=n_cells)

        return df

    def test_scvi_synthesis_basic(self, sample_expression_data):
        """Test basic scVI synthesis."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)

        n_samples = 50
        synthetic = gen.generate(
            data=sample_expression_data,
            n_samples=n_samples,
            method="scvi",
            target_col="cell_type",
            model_params={
                "epochs": 10,  # Low epochs for testing
                "n_latent": 5,
            },
        )

        assert synthetic is not None
        assert len(synthetic) == n_samples
        # Check that gene columns exist
        assert "gene_0" in synthetic.columns
        # Check that values are non-negative (expression data)
        gene_cols = [c for c in synthetic.columns if c.startswith("gene_")]
        assert (synthetic[gene_cols] >= 0).all().all()

    def test_scvi_synthesis_with_target_preservation(self, sample_expression_data):
        """Test that scVI preserves target column."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)

        synthetic = gen.generate(
            data=sample_expression_data,
            n_samples=30,
            method="scvi",
            target_col="cell_type",
            model_params={"epochs": 5, "n_latent": 5},
        )

        assert "cell_type" in synthetic.columns
        # Check that cell types are from original data
        original_types = set(sample_expression_data["cell_type"].unique())
        synthetic_types = set(synthetic["cell_type"].unique())
        assert synthetic_types.issubset(original_types)

    @pytest.mark.skipif(not scgen_available, reason="scgen not installed")
    def test_scgen_synthesis_basic(self, sample_expression_with_condition):
        """Test basic scGen synthesis."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)

        n_samples = 50
        synthetic = gen.generate(
            data=sample_expression_with_condition,
            n_samples=n_samples,
            method="scgen",
            target_col="cell_type",
            model_params={
                "epochs": 10,
                "n_latent": 5,
                "condition_col": "condition",
            },
        )

        assert synthetic is not None
        assert len(synthetic) == n_samples

    def test_scgen_fallback_to_scvi(self, sample_expression_data):
        """Test that scGen falls back to scVI when no condition_col is provided."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)

        # scGen without condition_col should fall back to scVI
        synthetic = gen.generate(
            data=sample_expression_data,
            n_samples=30,
            method="scgen",
            target_col="cell_type",
            model_params={"epochs": 5},
        )

        assert synthetic is not None
        assert len(synthetic) == 30

    def test_scvi_numeric_only_validation(self):
        """Test that scVI raises error for non-numeric data without target_col."""
        from calm_data_generator.generators.tabular import RealGenerator

        # Create data with only string columns
        df = pd.DataFrame(
            {
                "col1": ["a", "b", "c"] * 10,
                "col2": ["x", "y", "z"] * 10,
            }
        )

        gen = RealGenerator(auto_report=False)

        # RealGenerator returns None on failure and logs error
        result = gen.generate(
            data=df, n_samples=10, method="scvi", model_params={"epochs": 5}
        )
        assert result is None

    def test_scvi_model_params_kwargs(self, sample_expression_data):
        """Test that model_params are correctly passed to scVI."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)

        # Test with custom n_latent and n_layers
        synthetic = gen.generate(
            data=sample_expression_data,
            n_samples=20,
            method="scvi",
            model_params={
                "epochs": 5,
                "n_latent": 8,
                "n_layers": 2,
            },
        )

        assert synthetic is not None
        assert len(synthetic) == 20


class TestMethodValidation:
    """Test that new methods are properly validated."""

    def test_scvi_in_valid_methods(self):
        """Test that 'scvi' is a valid method."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)
        # Should not raise
        gen._validate_method("scvi")

    def test_scgen_in_valid_methods(self):
        """Test that 'scgen' is a valid method."""
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator(auto_report=False)
        # Should not raise
        gen._validate_method("scgen")
