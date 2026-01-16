"""
Generator Factory for CalmOps Synthetic Data Generators

This module provides a factory pattern for creating various synthetic data generators from the River library.
It standardizes the configuration and instantiation of these generators, making them easy to use
throughout the CalmOps project.

Key Features:
- **Centralized Generator Creation**: A single point of entry (`GeneratorFactory.create_generator`)
  to get instances of different River generators.
- **Standardized Configuration**: Uses a `GeneratorConfig` dataclass to manage all possible
  parameters for the supported generators, with clear aliases (e.g., `seed` and `random_state`).
- **Type Safety**: Employs a `GeneratorType` Enum to prevent errors from using incorrect generator names.
- **Preset Drift Scenarios**: Includes a helper method (`create_preset_concept_drift`) to quickly
  set up pairs of generators for concept drift experiments.
- **Discoverability**: Provides methods to list available generators (`get_available_generators`) and get
  information about their properties (`get_generator_info`).
"""

import warnings
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
# from river.datasets import synth  <-- Lazy loaded now

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class GeneratorType(Enum):
    """Enumeration of available River generator types."""

    AGRAWAL = "agrawal"
    SEA = "sea"
    HYPERPLANE = "hyperplane"
    RANDOM_TREE = "random_tree"
    STAGGER = "stagger"
    SINE = "sine"
    MIXED = "mixed"
    FRIEDMAN = "friedman"
    RANDOM_RBF = "random_rbf"


@dataclass
class GeneratorConfig:
    """Configuration class for all supported River generator parameters."""

    # Common parameters
    random_state: Optional[int] = None
    seed: Optional[int] = None  # Alias for random_state

    # Agrawal specific
    classification_function: int = 0
    balance_classes: bool = True
    perturbation: float = 0.0

    # SEA specific
    function: int = 0
    noise_percentage: float = 0.1

    # Hyperplane specific
    n_features: int = 10
    n_dims: Optional[int] = None  # Alias for n_features
    mag_change: float = 0.0
    noise_percentage_hyperplane: float = 0.05
    sigma: float = 0.1

    # Random Tree specific
    n_num_features: int = 5
    n_cat_features: int = 5
    n_categories_per_cat_feature: int = 5
    max_tree_depth: int = 5
    first_leaf_label: int = 1

    # Stagger specific
    classification_function_stagger: int = 0
    balance_classes_stagger: bool = True

    # Sine specific
    has_noise: bool = False
    noise_percentage_sine: float = 0.1

    # Mixed specific
    classification_function_mixed: int = 0
    balance_classes_mixed: bool = True

    # Friedman specific
    n_features_friedman: int = 10

    # Random RBF specific
    n_features_rbf: int = 10
    n_centroids: int = 50

    def __post_init__(self):
        """Post-initialization to handle aliases and ensure consistency."""
        # Handle seed/random_state alias
        if self.seed is not None and self.random_state is None:
            self.random_state = self.seed
        elif self.random_state is not None and self.seed is None:
            self.seed = self.random_state

        # Handle n_dims/n_features alias for hyperplane
        if self.n_dims is not None and self.n_features == 10:  # 10 is default
            self.n_features = self.n_dims
        elif self.n_features != 10 and self.n_dims is None:
            self.n_dims = self.n_features


class GeneratorFactory:
    """A factory class for creating River synthetic data generators."""

    @staticmethod
    def get_available_generators() -> List[GeneratorType]:
        """Returns a list of all available generator types."""
        return list(GeneratorType)

    @staticmethod
    def _get_river_synth():
        """Lazy loads river.datasets.synth."""
        try:
            from river.datasets import synth

            return synth
        except ImportError:
            raise ImportError(
                "The 'river' library is required for this generator. Please install it using 'pip install river'."
            )

    @staticmethod
    def create_generator(generator_type: GeneratorType, config: GeneratorConfig):
        """
        Creates a generator instance based on the specified type and configuration.

        Args:
            generator_type (GeneratorType): The type of generator to create.
            config (GeneratorConfig): A configuration object with all necessary parameters.

        Returns:
            An instance of a River generator, ready to be used.
        """
        generators = {
            GeneratorType.AGRAWAL: GeneratorFactory._create_agrawal,
            GeneratorType.SEA: GeneratorFactory._create_sea,
            GeneratorType.HYPERPLANE: GeneratorFactory._create_hyperplane,
            GeneratorType.RANDOM_TREE: GeneratorFactory._create_random_tree,
            GeneratorType.STAGGER: GeneratorFactory._create_stagger,
            GeneratorType.SINE: GeneratorFactory._create_sine,
            GeneratorType.MIXED: GeneratorFactory._create_mixed,
            GeneratorType.FRIEDMAN: GeneratorFactory._create_friedman,
            GeneratorType.RANDOM_RBF: GeneratorFactory._create_random_rbf,
        }

        if generator_type not in generators:
            raise ValueError(f"Unknown generator type: {generator_type}")

        return generators[generator_type](config)

    @staticmethod
    def _create_agrawal(config: GeneratorConfig):
        """Creates an Agrawal generator instance."""
        params = {
            "classification_function": config.classification_function,
            "balance_classes": config.balance_classes,
            "perturbation": config.perturbation,
        }
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.Agrawal(**params)

    @staticmethod
    def _create_sea(config: GeneratorConfig):
        """Creates a SEA generator instance."""
        params = {"variant": config.function, "noise": config.noise_percentage}
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.SEA(**params)

    @staticmethod
    def _create_hyperplane(config: GeneratorConfig):
        """Creates a Hyperplane generator instance."""
        params = {
            "n_features": config.n_features,
            "mag_change": config.mag_change,
            "sigma": config.sigma,
        }
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.Hyperplane(**params)

    @staticmethod
    def _create_random_tree(config: GeneratorConfig):
        """Creates a RandomTree generator instance."""
        params = {
            "n_num_features": config.n_num_features,
            "n_cat_features": config.n_cat_features,
            "n_categories_per_feature": config.n_categories_per_cat_feature,
            "max_tree_depth": config.max_tree_depth,
            "first_leaf_level": config.first_leaf_label,
        }
        if config.random_state is not None:
            params["seed_tree"] = config.random_state
            params["seed_sample"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.RandomTree(**params)

    @staticmethod
    def _create_stagger(config: GeneratorConfig):
        """Creates a STAGGER generator instance."""
        params = {
            "classification_function": config.classification_function_stagger,
            "balance_classes": config.balance_classes_stagger,
        }
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.STAGGER(**params)

    @staticmethod
    def _create_sine(config: GeneratorConfig):
        """Creates a Sine generator instance."""
        params = {"has_noise": config.has_noise}
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.Sine(**params)

    @staticmethod
    def _create_mixed(config: GeneratorConfig):
        """Creates a Mixed generator instance."""
        params = {
            "classification_function": config.classification_function_mixed,
            "balance_classes": config.balance_classes_mixed,
        }
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.Mixed(**params)

    @staticmethod
    def _create_friedman(config: GeneratorConfig):
        """Creates a Friedman generator instance."""
        params = {"n_features": config.n_features_friedman}
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.Friedman(**params)

    @staticmethod
    def _create_random_rbf(config: GeneratorConfig):
        """Creates a RandomRBF generator instance."""
        params = {
            "n_features": config.n_features_rbf,
            "n_centroids": config.n_centroids,
        }
        if config.random_state is not None:
            params["seed"] = config.random_state
        synth = GeneratorFactory._get_river_synth()
        return synth.RandomRBF(**params)

    @staticmethod
    def create_preset_concept_drift(
        generator_type: GeneratorType, seed: int = 42, drift_magnitude: float = 0.5
    ):
        """
        Creates a pair of generators (base and drift) suitable for concept drift experiments.

        Args:
            generator_type (GeneratorType): The base generator type to use.
            seed (int): A random seed for reproducibility.
            drift_magnitude (float): The magnitude of the drift to introduce between the two generators.

        Returns:
            A tuple containing the (base_generator, drift_generator).
        """
        base_config = GeneratorConfig(random_state=seed)
        drift_config = GeneratorConfig(random_state=seed + 1)

        if generator_type == GeneratorType.AGRAWAL:
            drift_config.classification_function = 1
        elif generator_type == GeneratorType.SEA:
            drift_config.function = 1
            drift_config.noise_percentage = (
                base_config.noise_percentage + drift_magnitude * 0.1
            )
        elif generator_type == GeneratorType.HYPERPLANE:
            drift_config.mag_change = drift_magnitude
        elif generator_type == GeneratorType.STAGGER:
            drift_config.classification_function_stagger = 1

        base_gen = GeneratorFactory.create_generator(generator_type, base_config)
        drift_gen = GeneratorFactory.create_generator(generator_type, drift_config)

        return base_gen, drift_gen

    @staticmethod
    def get_generator_info(generator_type: GeneratorType) -> Dict[str, Any]:
        """Returns a dictionary with information about a specific generator type."""
        info = {
            GeneratorType.AGRAWAL: {
                "name": "Agrawal",
                "description": "A classic classification dataset with multiple concept-drifting classification functions.",
                "features": 9,
                "classes": 2,
                "drift_capable": True,
            },
            GeneratorType.SEA: {
                "name": "SEA Concepts",
                "description": "Streaming Ensemble Algorithm (SEA) concepts with abrupt drift and noise.",
                "features": 3,
                "classes": 2,
                "drift_capable": True,
            },
            GeneratorType.HYPERPLANE: {
                "name": "Hyperplane",
                "description": "A dataset where the classification boundary is a rotating hyperplane, simulating gradual concept drift.",
                "features": "configurable",
                "classes": 2,
                "drift_capable": True,
            },
            GeneratorType.RANDOM_TREE: {
                "name": "Random Tree",
                "description": "A stable classification dataset generated by a fixed random tree structure.",
                "features": "configurable",
                "classes": 2,
                "drift_capable": False,
            },
            GeneratorType.STAGGER: {
                "name": "STAGGER Concepts",
                "description": "A dataset with three predefined, abruptly-drifting concepts.",
                "features": 3,
                "classes": 2,
                "drift_capable": True,
            },
            GeneratorType.SINE: {
                "name": "Sine",
                "description": "A regression dataset based on a sine wave with optional noise.",
                "features": 2,
                "classes": "regression",
                "drift_capable": False,
            },
            GeneratorType.MIXED: {
                "name": "Mixed",
                "description": "A dataset with both boolean and numeric features, with two possible classification functions.",
                "features": 4,
                "classes": 2,
                "drift_capable": False,
            },
            GeneratorType.FRIEDMAN: {
                "name": "Friedman",
                "description": "A synthetic regression dataset based on the Friedman #1 function.",
                "features": "configurable",
                "classes": "regression",
                "drift_capable": False,
            },
            GeneratorType.RANDOM_RBF: {
                "name": "Random RBF",
                "description": "A classification dataset generated using random Radial Basis Function (RBF) centers.",
                "features": "configurable",
                "classes": "variable",
                "drift_capable": False,
            },
        }
        return info.get(
            generator_type, {"name": "Unknown", "description": "Unknown generator"}
        )
