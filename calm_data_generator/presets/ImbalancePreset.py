import pandas as pd
import numpy as np
from typing import Dict, Union, Any, Optional, List
from .base import GeneratorPreset
from calm_data_generator.generators.tabular import RealGenerator


class ImbalancedGeneratorPreset(GeneratorPreset):
    """
    Preset designed to generate synthetic data with a specific imbalanced distribution.

    Useful for creating test datasets for drift detection or bias analysis.
    Uses 'resample' or generative methods with forced custom distributions on the target.
    """

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        imbalance_ratio: float = 0.1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generates imbalanced data.

        Args:
            target_col: The column to imbalance.
            imbalance_ratio: The ratio of the minority class (0.0 to 1.0).
                             e.g., 0.1 means minority class will be 10% of data.
        """
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", False), random_state=self.random_state
        )

        # Infer minority/majority classes if not provided
        # This is a simplification; for complex cases user should pass custom_distributions dict directly
        unique_vals = data[target_col].unique()
        if len(unique_vals) != 2:
            raise ValueError(
                "ImbalancedGeneratorPreset currently supports binary targets only."
            )

        # Arbitrarily pick one as minority if not specified (or use logic)
        minority_class = unique_vals[1]
        majority_class = unique_vals[0]

        custom_dist = {
            target_col: {
                minority_class: imbalance_ratio,
                majority_class: 1.0 - imbalance_ratio,
            }
        }

        # Merge with user provided custom_distributions if any
        user_dists = kwargs.pop("custom_distributions", {})
        custom_dist.update(user_dists)

        if self.verbose:
            print(
                f"[ImbalancedGeneratorPreset] Generating imbalanced data (ratio {imbalance_ratio}) for '{target_col}'..."
            )

        if self.verbose:
            print(
                f"[ImbalancedGeneratorPreset] Generating imbalanced data (ratio {imbalance_ratio}) for '{target_col}'..."
            )

        # Enforce CTGAN for this preset
        # Use minimal epochs if fast_dev_run is True
        epochs = 1 if self.fast_dev_run else 300

        config = {"method": "ctgan", "epochs": epochs}

        return gen.generate(
            data=data,
            n_samples=n_samples,
            target_col=target_col,
            custom_distributions=custom_dist,
            **config,
        )
