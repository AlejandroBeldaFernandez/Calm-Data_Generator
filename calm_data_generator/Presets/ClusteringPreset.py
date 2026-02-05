from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class GMMPreset(BasePreset):
    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="gmm",
            n_components=10,
            covariance_type="full",
            max_iter=100,
            n_init=10,
        )
