from .base import GeneratorPreset
from calm_data_generator.generators.tabular import RealGenerator


class CopulaPreset(GeneratorPreset):
    """
    Uses Gaussian Copula to model dependencies.
    Very fast and statistically robust baseline, though supports privacy less than GANs.
    """

    def generate(self, data, n_samples, **kwargs):
        gen = RealGenerator(
            auto_report=kwargs.pop("auto_report", True), random_state=self.random_state
        )

        if self.verbose:
            print("[CopulaPreset] Generating data using Gaussian Copula...")

        return gen.generate(
            data=data,
            n_samples=n_samples,
            method="copula",
            # Copula is fast, minimal params needed usually.
        )
