import pandas as pd
import numpy as np
from typing import Dict, Union, Any, Optional, List
from .base import GeneratorPreset
from calm_data_generator.generators.tabular import RealGenerator
from calm_data_generator.generators.clinical import ClinicalDataGenerator


class HighFidelityPreset(GeneratorPreset):
    """
    Preset optimized for maximum data quality and fidelity.

    Uses CTGAN with a high number of epochs and optimized batch size,
    prioritizing quality over speed. Forces adversarial validation.
    """

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        auto_report: bool = True,
    ) -> pd.DataFrame:
        gen = RealGenerator(auto_report=auto_report, random_state=self.random_state)

        # Default high-fidelity configuration
        # If fast_dev_run is True, use minimal settings for testing
        epochs = 1 if self.fast_dev_run else 1000

        config = {
            "method": "ctgan",
            "epochs": epochs,
            "batch_size": 250,
            "adversarial_validation": True,  # Ensure quality check
        }

        if self.verbose:
            print(
                f"[HighFidelityPreset] Generating data with high-fidelity settings (epochs={epochs})..."
            )

        return gen.generate(data=data, n_samples=n_samples, **config)


class SingleCellQualityPreset(GeneratorPreset):
    """
    Preset optimized for generating high-quality single-cell transcriptomics data (RNA-seq).

    Uses scVI (Single-cell Variational Inference) which is state-of-the-art for this task.
    """

    def generate(
        self,
        data: Any,
        n_samples: int,
        auto_report: bool = True,
    ) -> pd.DataFrame:
        gen = RealGenerator(auto_report=auto_report, random_state=self.random_state)

        # Configuration optimized for single-cell
        epochs = 1 if self.fast_dev_run else 400

        config = {
            "method": "scvi",
            "epochs": epochs,  # scVI converges well, 400 is often sufficient/good
            "n_latent": 10,  # Typical latent dimension for scVI
        }

        if self.verbose:
            print(
                f"[SingleCellQualityPreset] Generating single-cell data using scVI (epochs={epochs})..."
            )

        return gen.generate(data=data, n_samples=n_samples, **config)
