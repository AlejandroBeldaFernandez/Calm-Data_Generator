from .base import GeneratorPreset
from calm_data_generator.generators.tabular import RealGenerator


class TimeSeriesPreset(GeneratorPreset):
    """
    Preset optimized for generating sequential or time-series data.
    Uses TimeGAN or TimeVAE to capture temporal dynamics.
    """

    def generate(self, data, n_samples, sequence_key, time_col, **kwargs):
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            print(
                f"[TimeSeriesPreset] Generating time-series data using TimeGAN (seq_key='{sequence_key}')..."
            )

        # TimeGAN is standard for temporal deep learning
        epochs = 1 if self.fast_dev_run else 500

        if self.verbose:
            print(
                f"[TimeSeriesPreset] Generating time-series data using TimeGAN (seq_key='{sequence_key}', epochs={epochs})..."
            )

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="timegan",
            sequence_key=sequence_key,
            time_col=time_col,  # Critical args for temporal data
            epochs=epochs,  # Reasonable default
            batch_size=100,
            # No kwargs passed
        )
