#!/usr/bin/env python3
"""
Base Generator - Abstract Base Class for Data Generators.

This module provides the BaseGenerator class, an abstract base class that defines
the common interface and shared functionality for all data generators in the library.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np

from calm_data_generator.logger import get_logger


class BaseGenerator(ABC):
    """
    Abstract Base Class for all data generators.

    This class defines the common initialization logic and interface that all
    generators (RealGenerator, StreamGenerator, ClinicalDataGenerator) should follow.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        auto_report: bool = True,
        minimal_report: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the BaseGenerator.

        Args:
            random_state: Seed for random number generation for reproducibility.
            auto_report: If True, automatically generates a report after synthesis.
            minimal_report: If True, generates minimal reports (faster, no correlations/PCA).
            logger: An external logger instance. If None, a default one is created.
        """
        self.random_state = random_state
        self.auto_report = auto_report
        self.minimal_report = minimal_report
        self.rng = np.random.default_rng(random_state)
        self.logger = logger if logger else get_logger(self.__class__.__name__)

    @staticmethod
    def resolve_output_dir(output_dir: Optional[str]) -> Optional[str]:
        """Ensures the output directory exists and returns its path."""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """
        Main method to generate synthetic data. Must be implemented by subclasses.
        """
        pass
