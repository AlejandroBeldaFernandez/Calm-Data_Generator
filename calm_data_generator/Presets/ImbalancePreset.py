from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class ImbalancePreset(BasePreset):
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
            target_col=target_col,
            epochs=500,
            batch_size=100,
            discriminator_dim=(256, 256, 256),
            generator_dim=(256, 256),
            discriminator_steps=1,
            discriminator_lr=2e-4,
            generator_lr=2e-4,
            pac=10,
            verbose=True,
        )
