from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class DoppelGANgerPreset(BasePreset):
    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="dgan",
            max_sequence_len=100,
            sample_len=10,
            batch_size=100,
            epochs=400,
        )
