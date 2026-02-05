from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class BigDatasetPreset(BasePreset):
    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="ctgan",
            epochs=100,
            batch_size=1000,
            discriminator_dim=(512, 512, 512),
            generator_dim=(512, 512, 512),
            pac=10,
            cuda=True,
            verbose=True,
        )
