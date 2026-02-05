import pandas as pd
import numpy as np
import anndata
from calm_data_generator.generators.tabular import RealGenerator


def test_anndata_support_scvi():
    """Test passing AnnData directly to scvi method."""
    # 1. Create dummy AnnData
    n_cells = 50
    n_genes = 20
    X = np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"cell_type": np.random.choice(["A", "B"], size=n_cells)})
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # 2. Run generator with AnnData
    gen = RealGenerator(auto_report=False)
    synthetic_df = gen.generate(
        data=adata,
        n_samples=30,
        method="scvi",
        target_col="cell_type",
        model_params={"epochs": 5, "n_latent": 5},
    )

    # 3. Assertions
    assert synthetic_df is not None
    assert len(synthetic_df) == 30
    assert "gene_0" in synthetic_df.columns
    assert "cell_type" in synthetic_df.columns
    assert synthetic_df["cell_type"].isin(["A", "B"]).all()
    # Check that it's a DataFrame
    assert isinstance(synthetic_df, pd.DataFrame)


def test_anndata_support_scgen():
    """Test passing AnnData directly to scgen method."""
    # 1. Create dummy AnnData
    n_cells = 50
    n_genes = 20
    X = np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(["A", "B"], size=n_cells),
            "batch": np.random.choice(["b1", "b2"], size=n_cells),
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # 2. Run generator with AnnData
    gen = RealGenerator(auto_report=False)
    synthetic_df = gen.generate(
        data=adata,
        n_samples=30,
        method="scgen",
        target_col="cell_type",
        epochs=5,
        n_latent=5,
        condition_col="batch",
    )

    # 3. Assertions
    assert synthetic_df is not None
    assert len(synthetic_df) == 30
    assert "batch" in synthetic_df.columns
    assert synthetic_df["batch"].isin(["b1", "b2"]).all()
