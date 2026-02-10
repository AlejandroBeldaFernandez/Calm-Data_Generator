from .base import GeneratorPreset
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector


class ScenarioInjectorPreset(GeneratorPreset):
    """
    Directly leverages the ScenarioInjector to apply defined complex scenarios
    to an existing dataset, without necessarily generating new samples from scratch (unless desired).
    """

    def generate(self, data, scenario_config, **kwargs):
        # This preset behaves a bit differently: it modifies input data based on scenario
        injector = ScenarioInjector(seed=self.random_state)

        if self.verbose:
            print("[ScenarioInjectorPreset] Applying scenario configuration...")

        # Use evolve_features or generic apply
        # Use evolve_features or generic apply
        return injector.evolve_features(
            df=data, evolution_config=scenario_config.get("evolve_features")
        )
