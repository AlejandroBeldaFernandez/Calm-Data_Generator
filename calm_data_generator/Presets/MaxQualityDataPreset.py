from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class CTGANCopulaPreset(BasePreset):
    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="ctgan_copula",
            default_distribution="gaussian_kde",
            epochs=300,
            batch_size=500,
            generator_dim=(256, 256, 256),
            discriminator_dim=(256, 256, 256),
            enforce_min_max_values=True,
        )
