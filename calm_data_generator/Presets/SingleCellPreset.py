from Presets.BasePreset import BasePreset
from generators.tabular import RealGenerator
from typing import Optional
import pandas as pd


class SCVIPreset(BasePreset):
    """
    Preset especializado en síntesis de datos biológicos/single-cell usando scVI.
    Ideal para mantener la estructura de la distribución de conteos.
    """

    def run(
        self, data: pd.DataFrame, n_samples: int, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        # Instanciamos el generador
        gen = RealGenerator(
            auto_report=self.auto_report, random_state=self.random_state
        )

        # Parámetros óptimos para scVI en un entorno de síntesis
        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="scvi",
            n_hidden=128,  # Neuronas por capa oculta
            n_layers=2,  # Número de capas ocultas
            n_latent=30,  # Dimensión del espacio latente
            max_epochs=400,  # Entrenamiento profundo
            batch_size=128,
            use_gpu=True,  # scVI se beneficia mucho de CUDA
            target_col=target_col,  # Usado a menudo como etiqueta de celda o condición
        )
