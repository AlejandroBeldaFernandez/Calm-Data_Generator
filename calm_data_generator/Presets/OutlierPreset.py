from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class RandomForestPreset(BasePreset):
    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="rf",
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
        )
