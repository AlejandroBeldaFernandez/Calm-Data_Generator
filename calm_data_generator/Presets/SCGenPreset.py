from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class SCGenPreset(BasePreset):
    """
    Preset para scGen optimizado para capturar las diferencias
    entre las distintas clases presentes en el target_col.
    """

    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="scgen",
            epochs=100,
            batch_size=32,
            # Arquitectura robusta para capturar la varianza del target
            hidden_layers=[256, 128, 64],
            target_col=target_col,  # scGen usa esto para 'aprender' la firma de cada clase
        )
