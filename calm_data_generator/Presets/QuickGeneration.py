from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class LightGBMPreset(BasePreset):
    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="lgbm",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            n_jobs=-1,
        )
