#!/usr/bin/env python3
"""
Real Data Generator - Advanced data synthesis for real datasets.

This module provides the RealGenerator class, which serves as a powerful tool for
synthesizing data that mimics the characteristics of a real-world dataset. It integrates
several synthesis methods, from classic statistical approaches to modern deep learning models.

Key Features:
- **Multiple Synthesis Methods**: Supports a variety of methods including:
  - `cart`: FCS-like method using Decision Trees.
  - `rf`: FCS-like method using Random Forests.
  - `lgbm`: FCS-like method using LightGBM.
  - `gmm`: Gaussian Mixture Models (for numeric data).
  - `ctgan`, `tvae`, `copula`: Advanced deep learning models from the Synthetic Data Vault (SDV) library.
  - `datasynth`: Correlated attribute mode synthesis from the DataSynthesizer library.
  - `resample`: Simple resampling with replacement.
- **Conditional Synthesis**: Can generate data that follows custom-defined distributions for specified columns.
- **Target Balancing**: Automatically balances the distribution of the target variable.
- **Date Injection**: Capable of adding a timestamp column with configurable start dates and steps.
- **Comprehensive Reporting**: Automatically generates a detailed quality report comparing the synthetic data to the original, including visualizations and statistical metrics.
"""

import logging
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict, Any, List
import os
import math
import tempfile


# SDV and DataSynthesizer imports are now lazy-loaded


# Model imports
# Custom logger and reporter
from calm_data_generator.logger import get_logger
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter
from calm_data_generator.generators.drift.DriftInjector import DriftInjector
from calm_data_generator.generators.dynamics.ScenarioInjector import ScenarioInjector
from calm_data_generator.generators.configs import DateConfig


# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RealGenerator:
    """
    A class for advanced data synthesis from a real dataset, offering multiple generation methods and detailed reporting.
    """

    def __init__(
        self,
        auto_report: bool = True,
        minimal_report: bool = False,
        logger: Optional[logging.Logger] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initializes the RealGenerator.

        Args:
            auto_report (bool): If True, automatically generates a quality report after synthesis.
            minimal_report (bool): If True, generates minimal reports (faster, no correlations/PCA).
            logger (Optional[logging.Logger]): An external logger instance. If None, a new one is created.
            random_state (Optional[int]): Seed for random number generation for reproducibility.
        """
        self.auto_report = auto_report
        self.minimal_report = minimal_report
        self.logger = logger if logger else get_logger("RealGenerator")
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.reporter = QualityReporter(minimal=minimal_report)
        self.synthesizer = None
        self.metadata = None

    def _get_model_params(
        self, method: str, user_params: Optional[Dict] = None
    ) -> Dict:
        """Merges user parameters with defaults based on the method."""
        defaults = {
            "cart_iterations": 10,
            "cart_min_samples_leaf": None,
            "rf_n_estimators": None,
            "rf_min_samples_leaf": None,
            "lgbm_n_estimators": None,
            "lgbm_learning_rate": None,
            "gmm_n_components": 5,
            "gmm_covariance_type": "full",
            "sdv_epochs": 300,
            "sdv_batch_size": 100,
            "ds_k": 5,
            "smote_neighbors": 5,
            "adasyn_neighbors": 5,
            "dp_epsilon": 1.0,
            "dp_delta": 1e-5,
            "par_epochs": 100,
            "sequence_index": None,
            "diffusion_steps": 50,
        }
        params = defaults.copy()
        if user_params:
            params.update(user_params)
        return params

    def _validate_method(self, method: str):
        """Validates the synthesis method."""
        valid_methods = [
            "cart",
            "rf",
            "lgbm",
            "gmm",
            "ctgan",
            "tvae",
            "copula",
            "datasynth",
            "resample",
            "smote",
            "adasyn",
            "dp",
            "par",
            "diffusion",
            "timegan",
            "dgan",
            "copula_temporal",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown synthesis method '{method}'. Valid methods are: {valid_methods}"
            )

    def _build_metadata(self, data: pd.DataFrame):
        """Builds SDV metadata from a DataFrame."""
        self.logger.info("Building SDV metadata...")
        try:
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError(
                "The 'sdv' library is required for this method. Please install it using 'pip install sdv'."
            )
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        return metadata

    def _get_synthesizer(
        self,
        method: str,
        sdv_epochs: int,
        sdv_batch_size: int,
    ):
        """Initializes and returns the appropriate SDV synthesizer based on the method."""
        try:
            from sdv.single_table import (
                CTGANSynthesizer,
                TVAESynthesizer,
                CopulaGANSynthesizer,
            )
        except ImportError:
            raise ImportError(
                "The 'sdv' library is required for deep learning methods. Please install it using 'pip install sdv'."
            )

        # metadata is built externally or stored
        if self.metadata is None:
            raise ValueError("Metadata must be built before getting synthesizer.")

        if method == "ctgan":
            return CTGANSynthesizer(
                metadata=self.metadata,
                epochs=sdv_epochs,
                batch_size=sdv_batch_size,
                verbose=True,
            )
        elif method == "tvae":
            return TVAESynthesizer(
                metadata=self.metadata,
                epochs=sdv_epochs,
                batch_size=sdv_batch_size,
            )
        elif method == "copula":
            return CopulaGANSynthesizer(
                metadata=self.metadata,
                epochs=sdv_epochs,
                batch_size=sdv_batch_size,
                verbose=True,
            )
        else:
            raise ValueError(f"No SDV synthesizer for method '{method}'")

    def _validate_custom_distributions(
        self, custom_distributions: Dict, data: pd.DataFrame
    ) -> Dict:
        """Validates and normalizes custom distribution dictionaries."""
        if not isinstance(custom_distributions, dict):
            raise TypeError("custom_distributions must be a dictionary.")
        validated_distributions = custom_distributions.copy()
        for col, dist in validated_distributions.items():
            if col not in data.columns:
                raise ValueError(
                    f"Column '{col}' specified in custom_distributions does not exist in the dataset."
                )
            if not isinstance(dist, dict):
                raise TypeError(
                    f"The distribution for column '{col}' must be a dictionary."
                )
            if not dist:
                self.logger.warning(
                    f"Distribution for column '{col}' is empty. It will be ignored."
                )
                continue
            if any(p < 0 for p in dist.values()):
                raise ValueError(f"Proportions for column '{col}' cannot be negative.")
            total_proportion = sum(dist.values())
            if not math.isclose(total_proportion, 1.0):
                self.logger.warning(
                    f"Proportions for column '{col}' do not sum to 1.0 (sum={total_proportion}). They will be normalized."
                )
                validated_distributions[col] = {
                    k: v / total_proportion for k, v in dist.items()
                }
        return validated_distributions

    def _synthesize_sdv(
        self,
        data: pd.DataFrame,
        n_samples: int,
        method: str,
        sdv_epochs: int,
        sdv_batch_size: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using an SDV model, with support for conditional sampling."""
        self.logger.info(f"Starting conditional SDV synthesis with method: {method}...")
        # Always rebuild/refit for stateless operation on new data.
        # (Optimization: could cache if data/method matches, but keeping it simple/safe first)
        self.metadata = self._build_metadata(data)
        self.synthesizer = self._get_synthesizer(method, sdv_epochs, sdv_batch_size)
        self.synthesizer.fit(data)

        if not custom_distributions:
            self.logger.info(
                "No custom distributions provided. Generating samples unconditionally."
            )
            return self.synthesizer.sample(num_rows=n_samples)
        self.logger.info(
            f"Applying custom distributions via conditional sampling: {custom_distributions}"
        )
        if len(custom_distributions.keys()) > 1:
            self.logger.warning(
                f"Multiple columns found in custom_distributions. Conditioning on first column: '{next(iter(custom_distributions))}'."
            )
        col_to_condition = (
            target_col
            if target_col and target_col in custom_distributions
            else next(iter(custom_distributions))
        )
        dist = custom_distributions[col_to_condition]
        remaining_samples = n_samples
        all_synth_parts = []
        for value, proportion in dist.items():
            num_rows_for_val = int(n_samples * proportion)
            if num_rows_for_val > 0 and remaining_samples > 0:
                num_rows_for_val = min(num_rows_for_val, remaining_samples)
                self.logger.info(
                    f"Generating {num_rows_for_val} samples for '{col_to_condition}' = '{value}'"
                )
                try:
                    from sdv.sampling import Condition

                    synth_part = self.synthesizer.sample_from_conditions(
                        conditions=[
                            Condition(
                                num_rows=num_rows_for_val,
                                column_values={col_to_condition: value},
                            )
                        ]
                    )
                    all_synth_parts.append(synth_part)
                    remaining_samples -= len(synth_part)
                except Exception as e:
                    self.logger.warning(
                        f"Could not generate conditional samples for {col_to_condition}='{value}': {e}"
                    )
        if remaining_samples > 0:
            self.logger.info(
                f"Generating {remaining_samples} remaining samples unconditionally."
            )
            all_synth_parts.append(self.synthesizer.sample(num_rows=remaining_samples))
        if not all_synth_parts:
            raise RuntimeError("Conditional synthesis failed to generate any data.")
        return (
            pd.concat(all_synth_parts, ignore_index=True)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )

    def _synthesize_resample(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data by resampling from the original dataset, with optional weighting."""
        self.logger.info("Starting synthesis by resampling...")
        if not custom_distributions:
            return data.sample(
                n=n_samples, replace=True, random_state=self.random_state
            )
        self.logger.info(
            f"Applying custom distributions via weighted resampling: {custom_distributions}"
        )
        self.logger.warning(
            "The 'resample' method with custom distributions changes proportions but does not generate new data."
        )
        col_to_condition = (
            target_col
            if target_col and target_col in custom_distributions
            else next(iter(custom_distributions))
        )
        dist = custom_distributions[col_to_condition]
        weights = pd.Series(0.0, index=data.index)
        for category, proportion in dist.items():
            weights[data[col_to_condition] == category] = proportion
        if weights.sum() == 0:
            self.logger.warning(
                "Weights are all zero. Falling back to uniform resampling."
            )
            return data.sample(
                n=n_samples, replace=True, random_state=self.random_state
            )
        return data.sample(
            n=n_samples, replace=True, random_state=self.random_state, weights=weights
        )

    def _synthesize_smote(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        n_neighbors: int,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using SMOTE (Synthetic Minority Over-sampling Technique)."""
        self.logger.info("Starting SMOTE synthesis...")
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImblearnPipeline
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for SMOTE. Please install it."
            )

        if not target_col:
            raise ValueError("target_col is required for SMOTE synthesis.")

        X = data.drop(columns=target_col)
        y = data[target_col]

        # Handle Categorical Features (Basic encoding, SMOTE-NC would be better but keeping simple for now)
        # Assuming numeric for basic SMOTE or user pre-processed.
        # For robustness, we check types.
        if not X.select_dtypes(exclude=np.number).empty:
            self.logger.warning(
                "Standard SMOTE does not handle categorical features well. Use SMOTE-NC or encode first."
            )

        # Determine strategy
        # SMOTE usually balances classes. If n_samples > len(data), we need to oversample specific classes.
        # But here we want 'n_samples' total output.
        # Standard SMOTE resamples the minority class.
        # We'll use a strategy that tries to match the requested n_samples distribution if custom_dist provided,
        # otherwise balanced.

        # Simplified approach: Use SMOTE to balance, then sample n_samples.
        try:
            smote = SMOTE(k_neighbors=n_neighbors, random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            data_res = pd.concat([X_res, y_res], axis=1)

            if len(data_res) < n_samples:
                # If balanced is still less than n_samples, sample with replacement
                return data_res.sample(
                    n=n_samples, replace=True, random_state=self.random_state
                )
            else:
                return data_res.sample(
                    n=n_samples, replace=False, random_state=self.random_state
                )

        except Exception as e:
            self.logger.error(f"SMOTE synthesis failed: {e}")
            raise e

    def _synthesize_adasyn(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: str,
        n_neighbors: int,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using ADASYN (Adaptive Synthetic Sampling)."""
        self.logger.info("Starting ADASYN synthesis...")
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for ADASYN. Please install it."
            )

        if not target_col:
            raise ValueError("target_col is required for ADASYN synthesis.")

        X = data.drop(columns=target_col)
        y = data[target_col]

        try:
            adasyn = ADASYN(n_neighbors=n_neighbors, random_state=self.random_state)
            X_res, y_res = adasyn.fit_resample(X, y)
            data_res = pd.concat([X_res, y_res], axis=1)

            if len(data_res) < n_samples:
                return data_res.sample(
                    n=n_samples, replace=True, random_state=self.random_state
                )
            else:
                return data_res.sample(
                    n=n_samples, replace=False, random_state=self.random_state
                )
        except Exception as e:
            self.logger.error(f"ADASYN synthesis failed: {e}")
            raise e

    def _synthesize_dp(
        self,
        data: pd.DataFrame,
        n_samples: int,
        epsilon: float,
        delta: float,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes data using Differential Privacy (via SmartNoise/SNSynth).
        """
        self.logger.info(
            f"Starting Differential Privacy synthesis (epsilon={epsilon})..."
        )
        try:
            from snsynth import Synthesizer
        except ImportError:
            # Try alternate import or raise
            try:
                from smartnoise.synthesizers import PATECTGAN
            except ImportError:
                raise ImportError(
                    "smartnoise-synth (snsynth) is required for DP synthesis."
                )

        try:
            # We'll use PATE-CTGAN or MWEM as default. Let's try PATE-CTGAN for deep learning quality.
            # SNSynth wrapper
            synth = Synthesizer.create("pate_ctgan", epsilon=epsilon, verbose=True)
            synth.fit(data, preprocessor_eps=epsilon / 2.0)  # Split budget
            return synth.sample(n_samples)
        except Exception as e:
            self.logger.error(f"DP synthesis failed: {e}")
            raise e

    def _synthesize_par(
        self,
        data: pd.DataFrame,
        n_samples: int,
        epochs: int,
        sequence_key: str,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes data using Probabilistic AutoRegressive (PAR) model for Time Series.
        Uses SDV's PARSynthesizer.
        """
        self.logger.info("Starting PAR (Time Series) synthesis...")
        try:
            from sdv.sequential import PARSynthesizer
            from sdv.metadata import SingleTableMetadata
            # Note: PAR requires MultiTableMetadata usually or different setup in SDV 1.0+
            # Actually SDV 1.0+ wraps PAR in 'Sequential' module but expects metadata compatible with it.
        except ImportError:
            raise ImportError("sdv is required for PAR synthesis.")

        if not sequence_key:
            raise ValueError(
                "sequence_index (column name for entity ID) is required for PAR synthesis."
            )

        try:
            # For PAR, we need to know the sequence index (Entity) and Time index (optional/inferred)
            # We assume metadata is built or we build it manually
            # SDV 1.0+ changed API significantly. PARSynthesizer takes metadata.

            metadata = self._build_metadata(data)
            # We need to update metadata to set sequence key info if not auto-detected
            # In SDV 1.0, SingleTableMetadata doesn't natively support sequence key for PAR in the same way
            # PAR expects 'sequence_key' in constructor or metadata.

            metadata.update_column(column_name=sequence_key, sdtype="id")

            # Simple PAR usage
            synthesizer = PARSynthesizer(
                metadata=metadata,
                context_columns=[],  # Can add if needed
                epochs=epochs,
                verbose=True,
            )

            synthesizer.fit(data)

            # For sampling, PAR generates sequences. n_samples usually means number of SEQUENCES (Entities)
            # not total rows.
            # We assume n_samples is number of entities to generate for now, or we approximation.
            # If n_samples is total rows, it's hard to control exactly with PAR.
            # We'll assume n_samples = num_sequences (num entities)

            synth_data = synthesizer.sample(num_sequences=n_samples)
            return synth_data

        except Exception as e:
            self.logger.error(f"PAR synthesis failed: {e}")
            raise e

    def _synthesize_timegan(
        self,
        data: pd.DataFrame,
        n_samples: int,
        sequence_key: str,
        epochs: int = 100,
        seq_len: int = 24,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using TimeGAN.
        Requires ydata-synthetic library.
        """
        self.logger.info(
            f"Starting TimeGAN synthesis (epochs={epochs}, seq_len={seq_len})..."
        )

        try:
            from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
            from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
        except ImportError:
            self.logger.warning("ydata-synthetic not available. Falling back to PAR.")
            return self._synthesize_par(
                data, n_samples, epochs, sequence_key, target_col
            )

        try:
            # Prepare data: TimeGAN expects 3D array (samples, timesteps, features)
            # Group by sequence_key and create sequences
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if sequence_key in numeric_cols:
                numeric_cols.remove(sequence_key)

            groups = data.groupby(sequence_key)
            sequences = []
            for _, group in groups:
                seq_data = group[numeric_cols].values
                if len(seq_data) >= seq_len:
                    # Take first seq_len rows
                    sequences.append(seq_data[:seq_len])
                elif len(seq_data) > 0:
                    # Pad with last value
                    padded = np.vstack(
                        [seq_data, np.tile(seq_data[-1], (seq_len - len(seq_data), 1))]
                    )
                    sequences.append(padded)

            if not sequences:
                raise ValueError("No valid sequences found in data.")

            X = np.array(sequences)

            # Model parameters
            gan_args = ModelParameters(
                batch_size=min(32, len(X)), lr=5e-4, noise_dim=32, layers_dim=128
            )

            train_args = TrainParameters(
                epochs=epochs, sequence_length=seq_len, number_sequences=len(X)
            )

            # Create and train synthesizer
            synth = TimeSeriesSynthesizer(
                modelname="timegan", model_parameters=gan_args
            )
            synth.fit(X, train_args, num_cols=numeric_cols)

            # Generate synthetic data
            n_sequences = max(1, n_samples // seq_len)
            synth_sequences = synth.sample(n_sequences)

            # Convert back to DataFrame
            all_rows = []
            for i, seq in enumerate(synth_sequences):
                for t, row in enumerate(seq):
                    row_dict = {col: row[j] for j, col in enumerate(numeric_cols)}
                    row_dict[sequence_key] = f"synth_{i}"
                    row_dict["timestep"] = t
                    all_rows.append(row_dict)

            synth_df = pd.DataFrame(all_rows)
            self.logger.info(
                f"TimeGAN synthesis complete. Generated {len(synth_df)} samples."
            )
            return synth_df

        except Exception as e:
            self.logger.error(f"TimeGAN synthesis failed: {e}")
            raise e

    def _synthesize_dgan(
        self,
        data: pd.DataFrame,
        n_samples: int,
        sequence_key: str,
        epochs: int = 100,
        seq_len: int = 24,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using DGAN (DoppelGANger-style).
        Requires ydata-synthetic library.
        """
        self.logger.info(f"Starting DGAN synthesis (epochs={epochs})...")

        try:
            from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
            from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
        except ImportError:
            self.logger.warning(
                "ydata-synthetic not available. Falling back to TimeGAN."
            )
            return self._synthesize_timegan(
                data, n_samples, sequence_key, epochs, seq_len, target_col
            )

        try:
            # Similar preprocessing as TimeGAN
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if sequence_key in numeric_cols:
                numeric_cols.remove(sequence_key)

            groups = data.groupby(sequence_key)
            sequences = []
            for _, group in groups:
                seq_data = group[numeric_cols].values
                if len(seq_data) >= seq_len:
                    sequences.append(seq_data[:seq_len])
                elif len(seq_data) > 0:
                    padded = np.vstack(
                        [seq_data, np.tile(seq_data[-1], (seq_len - len(seq_data), 1))]
                    )
                    sequences.append(padded)

            if not sequences:
                raise ValueError("No valid sequences found in data.")

            X = np.array(sequences)

            # Model parameters for DGAN
            gan_args = ModelParameters(
                batch_size=min(32, len(X)), lr=1e-4, noise_dim=64, layers_dim=256
            )

            train_args = TrainParameters(
                epochs=epochs, sequence_length=seq_len, number_sequences=len(X)
            )

            # Use doppelganger model
            synth = TimeSeriesSynthesizer(
                modelname="doppelganger", model_parameters=gan_args
            )
            synth.fit(X, train_args, num_cols=numeric_cols)

            # Generate
            n_sequences = max(1, n_samples // seq_len)
            synth_sequences = synth.sample(n_sequences)

            # Convert to DataFrame
            all_rows = []
            for i, seq in enumerate(synth_sequences):
                for t, row in enumerate(seq):
                    row_dict = {col: row[j] for j, col in enumerate(numeric_cols)}
                    row_dict[sequence_key] = f"synth_{i}"
                    row_dict["timestep"] = t
                    all_rows.append(row_dict)

            synth_df = pd.DataFrame(all_rows)
            self.logger.info(
                f"DGAN synthesis complete. Generated {len(synth_df)} samples."
            )
            return synth_df

        except Exception as e:
            self.logger.error(f"DGAN synthesis failed: {e}")
            raise e

    def _synthesize_copula_temporal(
        self,
        data: pd.DataFrame,
        n_samples: int,
        sequence_key: str,
        time_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Gaussian Copula with temporal correlations.
        Lighter alternative to GAN-based methods.
        """
        self.logger.info("Starting Temporal Copula synthesis...")

        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from scipy.stats import spearmanr
        except ImportError:
            raise ImportError("sdv is required for Copula synthesis.")

        try:
            # Sort by sequence and time if available
            if time_col and time_col in data.columns:
                data = data.sort_values([sequence_key, time_col])
            else:
                data = data.sort_values(sequence_key)

            # Create lag features to capture temporal dependencies
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if sequence_key in numeric_cols:
                numeric_cols.remove(sequence_key)
            if time_col and time_col in numeric_cols:
                numeric_cols.remove(time_col)

            df_with_lags = data.copy()

            # Add lag-1 features for each numeric column within each sequence
            for col in numeric_cols[:5]:  # Limit to 5 columns to avoid explosion
                df_with_lags[f"{col}_lag1"] = df_with_lags.groupby(sequence_key)[
                    col
                ].shift(1)

            # Fill NaN from lag with column mean
            df_with_lags = df_with_lags.fillna(df_with_lags.mean(numeric_only=True))

            # Build metadata
            metadata = self._build_metadata(df_with_lags)

            # Train Copula
            synthesizer = GaussianCopulaSynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
            )
            synthesizer.fit(df_with_lags)

            # Sample
            synth_df = synthesizer.sample(n_samples)

            # Remove lag columns from output
            lag_cols = [c for c in synth_df.columns if "_lag1" in c]
            synth_df = synth_df.drop(columns=lag_cols, errors="ignore")

            # Sort output by sequence
            if sequence_key in synth_df.columns:
                synth_df = synth_df.sort_values(sequence_key).reset_index(drop=True)

            self.logger.info(
                f"Temporal Copula synthesis complete. Generated {len(synth_df)} samples."
            )
            return synth_df

        except Exception as e:
            self.logger.error(f"Temporal Copula synthesis failed: {e}")
            raise e

    def _synthesize_diffusion(
        self,
        data: pd.DataFrame,
        n_samples: int,
        steps: int,
    ) -> pd.DataFrame:
        """
        Synthesizes data using Tabular Diffusion (simple DDPM-like approach).
        Uses PyTorch for a basic denoising diffusion implementation.
        """
        self.logger.info(f"Starting Tabular Diffusion synthesis ({steps} steps)...")

        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler, LabelEncoder
        except ImportError:
            self.logger.warning("PyTorch not available. Falling back to CTGAN.")
            return self._synthesize_sdv(data, n_samples, "ctgan", 300, 100)

        # Preprocess: encode categoricals and scale numerics
        df = data.copy()
        encoders = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        scaler = StandardScaler()
        X = scaler.fit_transform(df.values.astype(float))
        X_tensor = torch.tensor(X, dtype=torch.float32)

        n_features = X.shape[1]

        # Simple MLP Denoiser
        class SimpleDenoiser(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim + 1, 128),  # +1 for timestep
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, dim),
                )

            def forward(self, x, t):
                t_emb = t.unsqueeze(-1) / steps  # Normalize timestep
                return self.net(torch.cat([x, t_emb], dim=-1))

        model = SimpleDenoiser(n_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Beta schedule
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Training loop (simplified)
        model.train()
        epochs = min(100, steps)
        batch_size = min(64, len(X_tensor))

        for epoch in range(epochs):
            perm = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[perm[i : i + batch_size]]
                t = torch.randint(0, steps, (len(batch),))

                # Forward diffusion
                noise = torch.randn_like(batch)
                sqrt_alpha = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
                sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod[t]).unsqueeze(-1)
                noisy = sqrt_alpha * batch + sqrt_one_minus_alpha * noise

                # Predict noise
                pred_noise = model(noisy, t.float())
                loss = nn.functional.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Sampling (reverse diffusion)
        model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, n_features)
            for t in reversed(range(steps)):
                t_batch = torch.full((n_samples,), t, dtype=torch.float32)
                pred_noise = model(x, t_batch)

                alpha = alphas[t]
                alpha_cumprod = alphas_cumprod[t]
                beta = betas[t]

                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0

                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
                ) + torch.sqrt(beta) * noise

        # Inverse transform
        synth_array = scaler.inverse_transform(x.numpy())
        synth_df = pd.DataFrame(synth_array, columns=df.columns)

        # Decode categoricals
        for col, le in encoders.items():
            synth_df[col] = (
                synth_df[col].round().clip(0, len(le.classes_) - 1).astype(int)
            )
            synth_df[col] = le.inverse_transform(synth_df[col])

        # Restore numeric types
        for col in numeric_cols:
            if data[col].dtype in [np.int64, np.int32]:
                synth_df[col] = synth_df[col].round().astype(int)

        self.logger.info(
            f"Diffusion synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_gmm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        gmm_n_components: int,
        gmm_covariance_type: str,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using Gaussian Mixture Models. Only supports numeric data."""
        self.logger.info("Starting GMM synthesis...")
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise ImportError("scikit-learn is required for GMM synthesis.")

        non_numeric_cols = data.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols.empty:
            raise ValueError(
                f"The 'gmm' method only supports numeric data, but found non-numeric columns: {list(non_numeric_cols)}."
            )
        gmm = GaussianMixture(
            n_components=gmm_n_components,
            covariance_type=gmm_covariance_type,
            random_state=self.random_state,
        )
        gmm.fit(data)
        synth_data, _ = gmm.sample(n_samples)
        synth = pd.DataFrame(synth_data, columns=data.columns)

        # If the target is supposed to be classification, round the results
        if target_col and target_col in synth.columns:
            unique_values = data[target_col].nunique()
            if unique_values < 25 or (unique_values / len(data)) < 0.05:
                self.logger.info(
                    f"Rounding GMM results for target column '{target_col}' to nearest integer."
                )
                synth[target_col] = synth[target_col].round().astype(int)

        if custom_distributions:
            self.logger.warning(
                "Applying custom distributions to GMM output via post-processing. This may break learned correlations."
            )
            col_to_condition = (
                target_col
                if target_col and target_col in custom_distributions
                else next(iter(custom_distributions))
            )
            dist = custom_distributions[col_to_condition]
            n_synth_samples = len(synth)
            new_values = []
            for value, proportion in dist.items():
                count = int(n_synth_samples * proportion)
                new_values.extend([value] * count)
            if len(new_values) < n_synth_samples:
                new_values.extend(
                    [list(dist.keys())[-1]] * (n_synth_samples - len(new_values))
                )
            self.rng.shuffle(new_values)
            synth[col_to_condition] = new_values[:n_synth_samples]
        return synth

    def _synthesize_fcs_generic(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict],
        model_factory_func,
        method_name: str,
        iterations: int,
        target_col: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Generic helper for Fully Conditional Specification (FCS) synthesis.

        Args:
            data: Original dataframe.
            n_samples: Number of samples to generate.
            custom_distributions: Optional customs distribution dict.
            model_factory_func: Callable(is_classification: bool, model_params: dict) -> model instance.
            method_name: Name of the method for logging.
            iterations: Number of FCS iterations.
        """
        self.logger.info(f"Starting {method_name} (FCS-style) synthesis...")

        if custom_distributions:
            self.logger.warning(
                f"For '{method_name}' method, custom distributions are handled by resampling the training data."
            )

        # Prepare initial synthetic data (bootstrap)
        X_real = data.copy()

        # Ensure object columns are category for consistency
        for col in X_real.select_dtypes(include=["object"]).columns:
            X_real[col] = X_real[col].astype("category")

        # Initial random sample
        # OPTIMIZATION: Instead of pure random sample (which might miss rare categories),
        # we repeat the original dataset as many times as possible, then sample the rest.
        n_real = len(X_real)
        if n_samples > n_real:
            n_repeats = n_samples // n_real
            remainder = n_samples % n_real
            X_synth_list = [X_real] * n_repeats
            if remainder > 0:
                X_synth_list.append(
                    X_real.sample(
                        n=remainder, replace=False, random_state=self.random_state
                    )
                )
            X_synth = pd.concat(X_synth_list, ignore_index=True)
            # Shuffle to break order
            X_synth = X_synth.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)
        else:
            # If we need fewer samples than real data, standard sample is fine (or could be stratified)
            X_synth = X_real.sample(
                n=n_samples, replace=True, random_state=self.random_state
            ).reset_index(drop=True)

        # Align categories for LGBM/Categorical handling
        cat_cols = X_real.select_dtypes(include="category").columns
        for col in cat_cols:
            if X_real[col].dtype.name == "category":
                X_synth[col] = pd.Categorical(
                    X_synth[col], categories=X_real[col].cat.categories
                )

        try:
            for it in range(iterations):
                self.logger.info(f"{method_name} iteration {it + 1}/{iterations}")
                for col in X_real.columns:
                    y_real_train = X_real[col]
                    Xr_real_train = X_real.drop(columns=col)
                    Xs_synth = X_synth.drop(columns=col)

                    # Determine task type
                    is_classification = False
                    if not pd.api.types.is_numeric_dtype(y_real_train):
                        is_classification = True
                    # Heuristic for low-cardinality numeric targets (treat as class)
                    elif col == target_col:
                        unique_values = y_real_train.nunique()
                        if (
                            unique_values < 25
                            or (unique_values / len(y_real_train)) < 0.05
                        ):
                            is_classification = True

                    # Model instantiation via factory
                    model = model_factory_func(is_classification)

                    # Prepare training data (potentially applying custom distributions)
                    y_to_fit, X_to_fit = (y_real_train, Xr_real_train)
                    if custom_distributions and col in custom_distributions:
                        X_to_fit, y_to_fit = self._apply_resampling_strategy(
                            Xr_real_train,
                            y_real_train,
                            custom_distributions[col],
                            n_samples,
                        )

                    # Encode categorical features for sklearn-based models (CART/RF)
                    # LGBM handles categories natively, but generic sklearn trees do not.
                    # We check if the model is LGBM-like by class name string check or attribute.
                    is_lgbm = "LGBM" in model.__class__.__name__

                    if not is_lgbm:
                        # Sklearn encoding
                        X_to_fit = X_to_fit.copy()
                        Xs_synth_input = Xs_synth.copy()
                        for c in X_to_fit.select_dtypes(include=["category"]).columns:
                            # Ensure X_to_fit is definitely category (redundant safety)
                            if not isinstance(X_to_fit[c].dtype, pd.CategoricalDtype):
                                X_to_fit[c] = X_to_fit[c].astype("category")

                            # Ensure Xs_synth_input is cast to the SAME categories before encoding
                            # This fixes the AttributeError: Can only use .cat accessor...
                            try:
                                # Re-cast to match training categories exactly
                                Xs_synth_input[c] = Xs_synth_input[c].astype(
                                    X_to_fit[c].dtype
                                )
                            except Exception:
                                # Fallback: if categories don't match, force conversion
                                Xs_synth_input[c] = pd.Categorical(
                                    Xs_synth_input[c],
                                    categories=X_to_fit[c].cat.categories,
                                )

                            # Now safe to encode
                            X_to_fit[c] = X_to_fit[c].cat.codes
                            Xs_synth_input[c] = Xs_synth_input[c].cat.codes
                    else:
                        # LGBM input
                        Xs_synth_input = Xs_synth

                    try:
                        model.fit(X_to_fit, y_to_fit)
                    except ValueError as e:
                        if "Input contains NaN" in str(e):
                            raise ValueError(
                                f"The '{method_name}' method failed due to NaNs. Please pre-clean data."
                            ) from e
                        raise e

                    y_synth_pred = model.predict(Xs_synth_input)

                    # Restore categorical type if needed
                    if y_real_train.dtype.name == "category":
                        y_synth_pred = pd.Categorical(
                            y_synth_pred, categories=y_real_train.cat.categories
                        )

                    X_synth[col] = y_synth_pred
            return X_synth
        except Exception as e:
            self.logger.error(f"{method_name} synthesis failed: {e}", exc_info=True)
            return None

    def _synthesize_cart(
        self,
        data: pd.DataFrame,
        n_samples: int,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Decision Trees."""

        def model_factory(is_classification):
            try:
                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
            except ImportError:
                raise ImportError("scikit-learn is required for CART synthesis.")

            model_params = {"random_state": self.random_state}
            if self.cart_min_samples_leaf is not None:
                model_params["min_samples_leaf"] = self.cart_min_samples_leaf
            return (
                DecisionTreeClassifier(**model_params)
                if is_classification
                else DecisionTreeRegressor(**model_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "CART",
            self.cart_iterations,
        )

    def _apply_resampling_strategy(self, X, y, custom_dist, n_samples):
        """Applies over/under-sampling to match a custom distribution before model training."""
        try:
            original_counts = y.value_counts().to_dict()
            target_total_size = n_samples
            target_counts = {
                k: int(v * target_total_size) for k, v in custom_dist.items()
            }
            oversampling_strategy = {
                k: v for k, v in target_counts.items() if v > original_counts.get(k, 0)
            }
            undersampling_strategy = {
                k: v for k, v in target_counts.items() if v < original_counts.get(k, 0)
            }
            steps = []

            try:
                from imblearn.over_sampling import RandomOverSampler
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.pipeline import Pipeline as ImblearnPipeline
            except ImportError:
                # Fallback if imblearn not available? Or raise.
                self.logger.warning(
                    "imbalanced-learn not installed. Skipping resampling strategy."
                )
                return X, y

            if oversampling_strategy:
                steps.append(
                    (
                        "o",
                        RandomOverSampler(
                            sampling_strategy=oversampling_strategy,
                            random_state=self.random_state,
                        ),
                    )
                )
            if undersampling_strategy:
                steps.append(
                    (
                        "u",
                        RandomUnderSampler(
                            sampling_strategy=undersampling_strategy,
                            random_state=self.random_state,
                        ),
                    )
                )
            if not steps:
                return X, y
            pipeline = ImblearnPipeline(steps=steps)
            self.logger.info(
                f"Applying resampling pipeline to match distribution for column '{y.name}'."
            )
            X_res, y_res = pipeline.fit_resample(X, y)
            return X_res, y_res
        except Exception as e:
            self.logger.warning(
                f"Could not apply resampling strategy for column '{y.name}': {e}. Using original distribution."
            )
            return X, y

    def _synthesize_cart(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        min_samples_leaf: Optional[int] = None,
        iterations: int = 10,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Decision Trees."""

        def model_factory(is_classification):
            try:
                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
            except ImportError:
                raise ImportError("scikit-learn is required for CART synthesis.")

            model_params = {"random_state": self.random_state}
            if min_samples_leaf is not None:
                model_params["min_samples_leaf"] = min_samples_leaf
            return (
                DecisionTreeClassifier(**model_params)
                if is_classification
                else DecisionTreeRegressor(**model_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "CART",
            iterations,
            target_col,
        )

    def _synthesize_rf(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        n_estimators: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        iterations: int = 10,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Random Forests."""

        def model_factory(is_classification):
            try:
                from sklearn.ensemble import (
                    RandomForestRegressor,
                    RandomForestClassifier,
                )
            except ImportError:
                raise ImportError("scikit-learn is required for RF synthesis.")

            model_params = {"random_state": self.random_state, "n_jobs": 1}
            if n_estimators is not None:
                model_params["n_estimators"] = n_estimators
            if min_samples_leaf is not None:
                model_params["min_samples_leaf"] = min_samples_leaf
            return (
                RandomForestClassifier(**model_params)
                if is_classification
                else RandomForestRegressor(**model_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "RF",
            iterations,
            target_col,
        )

    def _synthesize_lgbm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        n_estimators: Optional[int] = None,
        learning_rate: Optional[float] = None,
        iterations: int = 10,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with LightGBM."""

        def model_factory(is_classification):
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("lightgbm is required for LGBM synthesis.")

            model_params = {
                "random_state": self.random_state,
                "n_jobs": 1,
                "verbose": -1,
            }
            if n_estimators is not None:
                model_params["n_estimators"] = n_estimators
            if learning_rate is not None:
                model_params["learning_rate"] = learning_rate
            return (
                lgb.LGBMClassifier(**model_params)
                if is_classification
                else lgb.LGBMRegressor(**model_params)
            )

        return self._synthesize_fcs_generic(
            data,
            n_samples,
            custom_distributions,
            model_factory,
            "LGBM",
            iterations,
            target_col,
        )

    def _synthesize_datasynth(
        self,
        data: pd.DataFrame,
        n_samples: int,
        k: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Synthesizes data using DataSynthesizer in correlated attribute mode.
        Uses a secure temporary directory to avoid issues with file paths.
        """
        self.logger.info("Starting DataSynthesizer synthesis...")

        # Use tempfile.TemporaryDirectory() to create a unique and secure directory.
        # The 'with' block ensures the directory and its contents are removed upon exit.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_csv_path = os.path.join(temp_dir, "temp_data.csv")
            temp_description_file = os.path.join(temp_dir, "description.json")

            try:
                # Lazy import DataSynthesizer
                try:
                    from DataSynthesizer.DataDescriber import DataDescriber
                    from DataSynthesizer.DataGenerator import DataGenerator
                except ImportError:
                    raise ImportError(
                        "The 'DataSynthesizer' library is required for this method. Please install it."
                    )
                # Save the original data to the secure temporary location.
                data.to_csv(temp_csv_path, index=False)

                # Describe the dataset
                describer = DataDescriber()
                describer.describe_dataset_in_correlated_attribute_mode(
                    dataset_file=temp_csv_path, k=k
                )

                # Save the DataDescriber object for the DataGenerator to read.
                describer.save_dataset_description_to_file(temp_description_file)

                # Generate the dataset
                generator = DataGenerator()
                synth = generator.generate_dataset_in_correlated_attribute_mode(
                    n_samples=n_samples, description_file=temp_description_file
                )

                # Apply custom distributions (Post-processing)
                if custom_distributions:
                    self.logger.warning(
                        "Applying custom distributions to DataSynthesizer output via post-processing."
                    )
                    col_to_condition = (
                        target_col
                        if target_col and target_col in custom_distributions
                        else next(iter(custom_distributions))
                    )
                    dist = custom_distributions[col_to_condition]
                    n_synth_samples = len(synth)

                    # Resample values to match custom distribution
                    new_values = []
                    for value, proportion in dist.items():
                        count = int(n_synth_samples * proportion)
                        new_values.extend([value] * count)

                    # Fill if necessary and shuffle
                    if len(new_values) < n_synth_samples:
                        new_values.extend(
                            [list(dist.keys())[-1]]
                            * (n_synth_samples - len(new_values))
                        )

                    self.rng.shuffle(new_values)
                    synth[col_to_condition] = new_values[:n_synth_samples]

                return synth

            except Exception as e:
                self.logger.error(f"Synthesis with method 'datasynth' failed: {e}")
                # The 'with' statement will handle cleanup, just re-raise the error.
                raise e

    def _inject_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
        date_start: Optional[str],
        date_every: int,
        date_step: Optional[Dict[str, int]],
    ) -> pd.DataFrame:
        """Injects a date column into the DataFrame with specified frequency and step."""
        if date_start is None:
            return df
        if not isinstance(date_every, int) or date_every <= 0:
            raise ValueError(f"date_every must be a positive integer, got {date_every}")
        step = date_step or {"days": 1}
        valid_keys = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "microseconds",
            "nanoseconds",
        }
        if set(step.keys()) - valid_keys:
            raise ValueError(f"Invalid date_step keys: {set(step.keys()) - valid_keys}")
        try:
            start_ts = pd.to_datetime(date_start)
        except Exception as e:
            raise ValueError(f"Invalid date_start '{date_start}': {e}") from e
        total = len(df)
        if total == 0:
            df[date_col] = pd.Series(dtype="datetime64[ns]")
            return df
        periods = (total + date_every - 1) // date_every
        anchors = [start_ts + pd.DateOffset(**step) * i for i in range(periods)]
        series = (
            pd.Series(anchors).repeat(date_every).iloc[:total].reset_index(drop=True)
        )
        if date_col not in df.columns:
            df.insert(0, date_col, series)
        else:
            df[date_col] = series
        self.logger.info(
            f"[RealGenerator] Injected date column '{date_col}' starting at {start_ts}."
        )
        return df

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        method: str = "cart",
        target_col: Optional[str] = None,
        block_column: Optional[str] = None,
        output_dir: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        date_config: Optional["DateConfig"] = None,
        # Legacy/Unpacked date args for compat (optional)
        date_start: Optional[str] = None,
        date_every: int = 1,
        date_step: Optional[Dict[str, int]] = None,
        date_col: str = "timestamp",
        # End legacy
        balance_target: bool = False,
        save_dataset: bool = False,
        drift_injection_config: Optional[List[Dict]] = None,
        dynamics_config: Optional[Dict] = None,
        model_params: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        The main public method to generate synthetic data.

        Args:
            data (pd.DataFrame): The real dataset to be synthesized.
            n_samples (int): The number of synthetic samples to generate.
            method (str): The synthesis method to use.
            target_col (Optional[str]): The name of the target variable column.
            block_column (Optional[str]): The name of the column defining data blocks.
            output_dir (Optional[str]): Directory to save the report and dataset. Optional if save_dataset is False.
            custom_distributions (Optional[Dict]): A dictionary to specify custom distributions for columns.
            date_config (Optional[DateConfig]): Configuration for date injection.
            balance_target (bool): If True, balances the distribution of the target column.
            save_dataset (bool): If True, saves the generated dataset to a CSV file.
            drift_injection_config (Optional[List[Dict]]): List of drift injection configurations.
            dynamics_config (Optional[Dict]): Configuration for dynamics injection (feature evolution, target construction).
            model_params (Optional[Dict[str, Any]]): A dictionary of hyperparameters for the chosen synthesis model.

        Returns:
            Optional[pd.DataFrame]: The generated synthetic DataFrame, or None if synthesis fails.
        """
        self._validate_method(method)
        params = self._get_model_params(method, model_params)
        self.logger.info(
            f"Starting generation of {n_samples} samples using method '{method}'..."
        )

        # Resolve Date Config
        if date_config is None and date_start is not None:
            # Construct from legacy args
            from calm_data_generator.generators.configs import DateConfig

            date_config = DateConfig(
                start_date=date_start,
                frequency=date_every,
                step=date_step,
                date_col=date_col,
            )

        if custom_distributions:
            custom_distributions = self._validate_custom_distributions(
                custom_distributions, data
            )
        if (
            balance_target
            and target_col
            and (custom_distributions is None or target_col not in custom_distributions)
        ):
            self.logger.info(
                f"'balance_target' is True. Generating balanced distribution for '{target_col}'."
            )
            target_classes = data[target_col].unique()
            custom_distributions = custom_distributions or {}
            custom_distributions[target_col] = {
                c: 1 / len(target_classes) for c in target_classes
            }
        try:
            synth = None
            if method in ["ctgan", "tvae", "copula"]:
                synth = self._synthesize_sdv(
                    data,
                    n_samples,
                    method,
                    params["sdv_epochs"],
                    params["sdv_batch_size"],
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                )
            elif method == "resample":
                synth = self._synthesize_resample(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                )
            elif method == "cart":
                synth = self._synthesize_cart(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    min_samples_leaf=params["cart_min_samples_leaf"],
                    iterations=params["cart_iterations"],
                )
            elif method == "rf":
                synth = self._synthesize_rf(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    n_estimators=params["rf_n_estimators"],
                    min_samples_leaf=params["rf_min_samples_leaf"],
                    iterations=params["cart_iterations"],
                )
            elif method == "lgbm":
                synth = self._synthesize_lgbm(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    n_estimators=params["lgbm_n_estimators"],
                    learning_rate=params["lgbm_learning_rate"],
                    iterations=params["cart_iterations"],
                )
            elif method == "gmm":
                synth = self._synthesize_gmm(
                    data,
                    n_samples,
                    gmm_n_components=params["gmm_n_components"],
                    gmm_covariance_type=params["gmm_covariance_type"],
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                )
                synth = self._synthesize_datasynth(
                    data,
                    n_samples,
                    k=params["ds_k"],
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                )
            elif method == "smote":
                synth = self._synthesize_smote(
                    data,
                    n_samples,
                    target_col=target_col,
                    n_neighbors=params["smote_neighbors"],
                    custom_distributions=custom_distributions,
                )
            elif method == "adasyn":
                synth = self._synthesize_adasyn(
                    data,
                    n_samples,
                    target_col=target_col,
                    n_neighbors=params["adasyn_neighbors"],
                    custom_distributions=custom_distributions,
                )
            elif method == "dp":
                synth = self._synthesize_dp(
                    data,
                    n_samples,
                    epsilon=params["dp_epsilon"],
                    delta=params["dp_delta"],
                    target_col=target_col,
                )
            elif method == "par":
                synth = self._synthesize_par(
                    data,
                    n_samples,
                    epochs=params["par_epochs"],
                    sequence_key=params.get("sequence_index")
                    or block_column,  # Use block_column as default sequence key
                    target_col=target_col,
                )
            elif method == "diffusion":
                synth = self._synthesize_diffusion(
                    data, n_samples, steps=params["diffusion_steps"]
                )
            elif method == "timegan":
                synth = self._synthesize_timegan(
                    data,
                    n_samples,
                    sequence_key=params.get("sequence_index") or block_column,
                    epochs=params.get("timegan_epochs", 100),
                    seq_len=params.get("seq_len", 24),
                    target_col=target_col,
                )
            elif method == "dgan":
                synth = self._synthesize_dgan(
                    data,
                    n_samples,
                    sequence_key=params.get("sequence_index") or block_column,
                    epochs=params.get("dgan_epochs", 100),
                    seq_len=params.get("seq_len", 24),
                    target_col=target_col,
                )
            elif method == "copula_temporal":
                synth = self._synthesize_copula_temporal(
                    data,
                    n_samples,
                    sequence_key=params.get("sequence_index") or block_column,
                    time_col=params.get("time_col"),
                    target_col=target_col,
                )

            # --- Constraints Application ---
            if synth is not None and constraints:
                self.logger.info(
                    f"Applying {len(constraints)} constraints to generated data..."
                )
                # Simple Rejection Sampling or Filtering
                # constraint format: {'col': 'Age', 'op': '>', 'val': 18}

                initial_count = len(synth)
                valid_mask = pd.Series(True, index=synth.index)

                for const in constraints:
                    col = const.get("col")
                    op = const.get("op")
                    val = const.get("val")

                    if col not in synth.columns:
                        self.logger.warning(
                            f"Constraint column '{col}' not found. Skipping."
                        )
                        continue

                    if op == ">":
                        valid_mask &= synth[col] > val
                    elif op == "<":
                        valid_mask &= synth[col] < val
                    elif op == ">=":
                        valid_mask &= synth[col] >= val
                    elif op == "<=":
                        valid_mask &= synth[col] <= val
                    elif op == "==":
                        valid_mask &= synth[col] == val
                    elif op == "!=":
                        valid_mask &= synth[col] != val

                synth = synth[valid_mask].reset_index(drop=True)
                final_count = len(synth)
                dropped = initial_count - final_count

                if dropped > 0:
                    self.logger.warning(
                        f"Constraints filtering dropped {dropped} rows ({dropped / initial_count:.1%}). consider loosening constraints or improving model."
                    )
                    # Optional: Retry loop could go here (generate more to compensate)
                    if len(synth) < n_samples:
                        self.logger.info(
                            "Regenerating to fill filtered rows not implemented yet in this pass."
                        )  # TODO: Add loop

            if synth is not None:
                self.logger.info(f"Successfully synthesized {len(synth)} samples.")

                # --- Dynamics Injection (Feature Evolution & Target Construction) ---
                if dynamics_config:
                    self.logger.info("Applying dynamics injection...")
                    # Initialize dynamics injector
                    dyn_injector = ScenarioInjector(seed=self.random_state)

                    # Evolve Features
                    if "evolve_features" in dynamics_config:
                        self.logger.info("Evolving features (Dynamics)...")
                        evolve_args = dynamics_config["evolve_features"]

                        # Inject dates early for dynamics if needed
                        if date_config and date_config.start_date:
                            synth = self._inject_dates(
                                df=synth,
                                date_col=date_config.date_col,
                                date_start=date_config.start_date,
                                date_every=date_config.frequency,
                                date_step=date_config.step,
                            )

                        time_col = date_config.date_col if date_config else "timestamp"
                        synth = dyn_injector.evolve_features(
                            synth, time_col=time_col, evolution_config=evolve_args
                        )

                    # Construct/Overwrite Target
                    if "construct_target" in dynamics_config:
                        self.logger.info("Constructing dynamic target (Dynamics)...")
                        target_args = dynamics_config["construct_target"]
                        synth = dyn_injector.construct_target(synth, **target_args)

                # --- Date Injection (if not done in dynamics) ---
                if date_config and date_config.start_date:
                    synth = self._inject_dates(
                        df=synth,
                        date_col=date_config.date_col,
                        date_start=date_config.start_date,
                        date_every=date_config.frequency,
                        date_step=date_config.step,
                    )

                # --- Drift Injection ---
                if drift_injection_config:
                    self.logger.info("Applying drift injection...")

                    # Resolve dir for drift injector
                    drift_out_dir = (
                        output_dir or "."
                    )  # Drift injector might need a dir, fallback to current
                    time_col_name = date_config.date_col if date_config else "timestamp"

                    drift_injector = DriftInjector(
                        original_df=synth,  # We drift the synthetic data
                        output_dir=drift_out_dir,
                        generator_name=f"{method}_drifted",
                        target_column=target_col,
                        block_column=block_column,
                        time_col=time_col_name,
                        random_state=self.random_state,
                    )

                    for drift_conf in drift_injection_config:
                        method_name = drift_conf.get("method")
                        params_drift = drift_conf.get("params", {})

                        if hasattr(drift_injector, method_name):
                            self.logger.info(f"Injecting drift: {method_name}")
                            drift_method = getattr(drift_injector, method_name)
                            try:
                                # Most DriftInjector methods return the modified DF
                                # We check if 'df' is in params, injecting it if needed.
                                if "df" not in params_drift:
                                    params_drift["df"] = synth

                                res = drift_method(**params_drift)
                                # Update synth if result is dataframe
                                if isinstance(res, pd.DataFrame):
                                    synth = res
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to apply drift {method_name}: {e}"
                                )
                                raise e
                        else:
                            self.logger.warning(
                                f"Drift method '{method_name}' not found in DriftInjector."
                            )

                if self.auto_report and output_dir:
                    time_col_name = date_config.date_col if date_config else "timestamp"
                    self.reporter.generate_comprehensive_report(
                        real_df=data,
                        synthetic_df=synth,
                        generator_name=f"RealGenerator_{method}",
                        output_dir=output_dir,
                        target_column=target_col,
                        time_col=time_col_name,
                    )

                # Save the generated dataset for inspection
                if save_dataset:  # Only save if save_dataset is True
                    if not output_dir:
                        raise ValueError(
                            "output_dir must be provided if save_dataset is True"
                        )
                    try:
                        save_path = os.path.join(
                            output_dir, f"synthetic_data_{method}.csv"
                        )
                        synth.to_csv(save_path, index=False)
                        self.logger.info(
                            f"Generated synthetic dataset saved to: {save_path}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to save synthetic dataset: {e}")

                return synth
            else:
                self.logger.error(
                    f"Synthesis method '{method}' failed to generate data."
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Synthesis with method '{method}' failed: {e}", exc_info=True
            )
            return None
