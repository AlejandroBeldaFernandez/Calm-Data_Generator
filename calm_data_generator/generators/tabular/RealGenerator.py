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
  - `ctgan`, `tvae`: Advanced deep learning/graph
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
from typing import Optional, Dict, Any, List, Union
import os
import math
import tempfile


# Synthcity and customized dependencies are lazy-loaded

# Model imports
# Custom logger and reporter
from calm_data_generator.generators.base import BaseGenerator
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter
from calm_data_generator.generators.configs import DateConfig, DriftConfig, ReportConfig
from calm_data_generator.generators.drift.DriftInjector import DriftInjector

# Synthcity import


# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RealGenerator(BaseGenerator):
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
        super().__init__(
            random_state=random_state,
            auto_report=auto_report,
            minimal_report=minimal_report,
            logger=logger,
        )
        self.reporter = QualityReporter(minimal=minimal_report)
        self.synthesizer = None
        self.metadata = None

    def _get_model_params(
        self, method: str, user_params: Optional[Dict] = None
    ) -> Dict:
        """Merges user parameters with defaults based on the method."""
        # Standard parameter names (matching sklearn/lightgbm/Synthcity APIs)
        defaults = {
            # FCS methods (CART, RF, LGBM)
            "iterations": 10,
            # GMM
            "n_components": 5,
            "covariance_type": "full",
            # Synthcity (CTGAN, TVAE)
            "epochs": 300,
            "batch_size": 100,
            # SMOTE/ADASYN
            "k_neighbors": 5,  # SMOTE
            "n_neighbors": 5,  # ADASYN
            # Time Series
            "sequence_key": None,
            # Diffusion
            "steps": 50,
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
            "copula",
            "ctgan",
            "tvae",
            "resample",
            "adasyn",
            "smote",
            "diffusion",
            "ddpm",
            "timegan",
            "timevae",
            "scvi",
            "gears",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown synthesis method '{method}'. Valid methods are: {valid_methods}"
            )

    def _get_synthesizer(
        self,
        method: str,
        **model_kwargs,
    ):
        """Initializes and returns the appropriate Synthcity plugin."""
        try:
            from synthcity.plugins import Plugins
        except ImportError:
            raise ImportError(
                "synthcity is required for this method. Please install it."
            )

        return Plugins().get(method, **model_kwargs)

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

    def _synthesize_ctgan(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using CTGAN via Synthcity."""
        self.logger.info("Starting CTGAN synthesis via Synthcity...")
        if "epochs" in model_kwargs:
            model_kwargs["n_iter"] = model_kwargs.pop("epochs")

        syn = self._get_synthesizer("ctgan", **model_kwargs)
        syn.fit(data)
        return syn.generate(count=n_samples).dataframe()

    def _synthesize_tvae(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using TVAE via Synthcity."""
        self.logger.info("Starting TVAE synthesis via Synthcity...")
        if "epochs" in model_kwargs:
            model_kwargs["n_iter"] = model_kwargs.pop("epochs")

        syn = self._get_synthesizer("tvae", **model_kwargs)
        syn.fit(data)
        return syn.generate(count=n_samples).dataframe()

    def _synthesize_copula(
        self,
        data: pd.DataFrame,
        n_samples: int,
        method: str,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **model_kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using Copulae (Gaussian Copula)."""
        self.logger.info("Starting synthesis using Gaussian Copula...")
        try:
            from copulae import GaussianCopula
            from sklearn.preprocessing import MinMaxScaler
        except ImportError:
            raise ImportError(
                "copulae and scikit-learn are required for the 'copula' method."
            )

        # Preprocessing: Copulas work on [0, 1] margins
        # We'll use a simple MinMax scaler for now to get to [0, 1],
        # but true copulas usually use empirical CDF transofmration.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("Copula synthesis requires at least some numeric columns.")

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data[numeric_cols])

        # Fit copula
        cop = GaussianCopula(dim=len(numeric_cols))
        cop.fit(X_scaled)

        # Sample
        samples = cop.random(n_samples)

        # Inverse transform
        synth_numeric = pd.DataFrame(
            scaler.inverse_transform(samples), columns=numeric_cols
        )

        # Handle non-numeric columns by simple resampling (naive approach for consistency)
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        if not non_numeric_cols.empty:
            self.logger.warning(
                "Copula method currently only models numeric correlations. Non-numeric columns will be resampled independently."
            )
            synth_non_numeric = (
                data[non_numeric_cols]
                .sample(n=n_samples, replace=True)
                .reset_index(drop=True)
            )
            synth = pd.concat([synth_numeric, synth_non_numeric], axis=1)
        else:
            synth = synth_numeric

        return synth[data.columns]  # Restore original order

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
        custom_distributions: Optional[Dict] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using SMOTE (Synthetic Minority Over-sampling Technique)."""
        self.logger.info("Starting SMOTE synthesis...")
        try:
            from imblearn.over_sampling import SMOTE
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
            k_neighbors = kwargs.get("k_neighbors", 5)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
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
        custom_distributions: Optional[Dict] = None,
        **kwargs,
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
            n_neighbors = kwargs.get("n_neighbors", 5)
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

    # REMOVED: _synthesize_timegan and _synthesize_dgan methods
    # These methods required ydata-synthetic library which is not used in this project.
    # For time series synthesis, use Synthcity's time series models or other alternatives.

    def _synthesize_diffusion(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes data using Tabular Diffusion (simple DDPM-like approach).
        Uses PyTorch for a basic denoising diffusion implementation.
        """
        steps = kwargs.get("steps", 1000)  # Retrieve steps from kwargs
        self.logger.info(f"Starting Tabular Diffusion synthesis ({steps} steps)...")

        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler, LabelEncoder
        except ImportError:
            self.logger.warning("PyTorch not available. Falling back to CTGAN.")
            return self._synthesize_synthcity(data, n_samples, "ctgan", 300, 100)

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

    def _synthesize_ddpm(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes data using Synthcity's TabDDPM (Tabular Denoising Diffusion).

        TabDDPM is a more advanced diffusion model specifically designed for tabular data.
        It supports multiple architectures (MLP, ResNet, TabNet) and advanced schedulers.

        Args:
            data: Input DataFrame
            n_samples: Number of samples to generate
            **kwargs: Additional parameters for TabDDPM:
                - n_iter: int = 1000 - Training epochs
                - lr: float = 0.002 - Learning rate
                - batch_size: int = 1024 - Batch size
                - num_timesteps: int = 1000 - Diffusion timesteps
                - model_type: str = "mlp" - Model architecture ("mlp", "resnet", "tabnet")
                - scheduler: str = "cosine" - Beta scheduler ("cosine", "linear")
                - gaussian_loss_type: str = "mse" - Loss type ("mse", "kl")
                - is_classification: bool = False - Whether task is classification

        Returns:
            Synthetic DataFrame
        """
        self.logger.info("Starting TabDDPM synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
        except ImportError:
            self.logger.warning(
                "Synthcity not available. Falling back to custom diffusion."
            )
            return self._synthesize_diffusion(data, n_samples, **kwargs)

        # Extract DDPM-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        lr = kwargs.get("lr", 0.002)
        batch_size = kwargs.get("batch_size", 1024)
        num_timesteps = kwargs.get("num_timesteps", 1000)
        model_type = kwargs.get("model_type", "mlp")
        scheduler = kwargs.get("scheduler", "cosine")
        gaussian_loss_type = kwargs.get("gaussian_loss_type", "mse")
        is_classification = kwargs.get("is_classification", False)

        # Load plugin
        plugin = Plugins().get(
            "ddpm",
            n_iter=n_iter,
            lr=lr,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            model_type=model_type,
            scheduler=scheduler,
            gaussian_loss_type=gaussian_loss_type,
            is_classification=is_classification,
        )

        # Prepare data
        loader = GenericDataLoader(data)

        # Train
        self.logger.info(f"Training TabDDPM for {n_iter} epochs...")
        plugin.fit(loader)

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic samples...")
        synth = plugin.generate(count=n_samples)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TabDDPM synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_timegan(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Synthcity's TimeGAN.

        TimeGAN is designed for sequential/temporal data with multiple entities.
        It learns both temporal dynamics and feature distributions.

        Args:
            data: Input DataFrame with temporal structure
            n_samples: Number of sequences to generate
            **kwargs: Additional parameters for TimeGAN:
                - n_iter: int = 1000 - Training epochs
                - n_units_hidden: int = 100 - Hidden units
                - batch_size: int = 128 - Batch size
                - lr: float = 0.001 - Learning rate

        Returns:
            Synthetic DataFrame with temporal structure
        """
        self.logger.info("Starting TimeGAN synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            self.logger.error("Synthcity not available for TimeGAN.")
            raise ImportError(
                "TimeGAN requires synthcity. Install with: pip install synthcity"
            )

        # Extract TimeGAN-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        n_units_hidden = kwargs.get("n_units_hidden", 100)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        # Load plugin
        plugin = Plugins().get(
            "timegan",
            n_iter=n_iter,
            n_units_hidden=n_units_hidden,
            batch_size=batch_size,
            lr=lr,
        )

        # Prepare time series data
        # TimeSeriesDataLoader expects data in specific format
        loader = TimeSeriesDataLoader(data)

        # Train
        self.logger.info(f"Training TimeGAN for {n_iter} epochs...")
        plugin.fit(loader)

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TimeGAN synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_timevae(
        self, data: pd.DataFrame, n_samples: int, **kwargs
    ) -> pd.DataFrame:
        """
        Synthesizes time series data using Synthcity's TimeVAE.

        TimeVAE is a variational autoencoder designed for temporal data.
        It's generally faster than TimeGAN and works well for regular time series.

        Args:
            data: Input DataFrame with temporal structure
            n_samples: Number of sequences to generate
            **kwargs: Additional parameters for TimeVAE:
                - n_iter: int = 1000 - Training epochs
                - decoder_n_layers_hidden: int = 2 - Decoder layers
                - decoder_n_units_hidden: int = 100 - Decoder units
                - batch_size: int = 128 - Batch size
                - lr: float = 0.001 - Learning rate

        Returns:
            Synthetic DataFrame with temporal structure
        """
        self.logger.info("Starting TimeVAE synthesis (Synthcity)...")

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
        except ImportError:
            self.logger.error("Synthcity not available for TimeVAE.")
            raise ImportError(
                "TimeVAE requires synthcity. Install with: pip install synthcity"
            )

        # Extract TimeVAE-specific parameters
        n_iter = kwargs.get("n_iter", kwargs.get("epochs", 1000))
        decoder_n_layers_hidden = kwargs.get("decoder_n_layers_hidden", 2)
        decoder_n_units_hidden = kwargs.get("decoder_n_units_hidden", 100)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("lr", 0.001)

        # Load plugin
        plugin = Plugins().get(
            "timevae",
            n_iter=n_iter,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            batch_size=batch_size,
            lr=lr,
        )

        # Prepare time series data
        loader = TimeSeriesDataLoader(data)

        # Train
        self.logger.info(f"Training TimeVAE for {n_iter} epochs...")
        plugin.fit(loader)

        # Generate
        self.logger.info(f"Generating {n_samples} synthetic sequences...")
        synth = plugin.generate(count=n_samples)
        synth_df = synth.dataframe()

        self.logger.info(
            f"TimeVAE synthesis complete. Generated {len(synth_df)} samples."
        )
        return synth_df

    def _synthesize_gmm(
        self,
        data: pd.DataFrame,
        n_samples: int,
        target_col: Optional[str] = None,
        custom_distributions: Optional[Dict] = None,
        **kwargs,
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
        model_params = {
            "n_components": kwargs.get("n_components", 5),
            "covariance_type": kwargs.get("covariance_type", "full"),
            "random_state": self.random_state,
        }
        # Filter kwargs to only include what GaussianMixture expects if necessary,
        # but for now let's just update with what's provided.
        model_p = model_params.copy()
        model_p.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in ["n_components", "covariance_type"]
            }
        )

        gmm = GaussianMixture(**model_p)
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

    def _synthesize_scvi(
        self,
        data: Union[pd.DataFrame, Any],
        n_samples: int,
        target_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes single-cell-like data using scVI (Variational Autoencoder).

        This method treats the input as a gene expression matrix where:
        - Rows are cells/samples
        - Columns are genes/features

        Args:
            data: DataFrame or AnnData with numeric expression values
            n_samples: Number of synthetic samples to generate
            target_col: Optional column to preserve as metadata (will be excluded from training)
            **kwargs: Additional parameters passed to scVI model (n_latent, n_layers, etc.)

        Returns:
            DataFrame with synthetic samples
        """
        self.logger.info("Starting scVI synthesis for single-cell data...")
        print(f"DEBUG: Entering _synthesize_scvi with data type: {type(data)}")

        try:
            import anndata
            import scvi
        except ImportError:
            raise ImportError(
                "scvi-tools and anndata are required for scVI synthesis. "
                "Install with: pip install scvi-tools anndata"
            )

        # Create or use AnnData object
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            adata = data
            # Ensure target_col is in obs if provided
            if target_col and target_col not in adata.obs.columns:
                self.logger.warning(
                    f"target_col '{target_col}' not found in AnnData.obs."
                )
        else:
            # Separate metadata from expression data
            metadata_cols = []
            if target_col and target_col in data.columns:
                metadata_cols.append(target_col)

            # Get expression columns (numeric only, excluding metadata)
            expr_cols = [c for c in data.columns if c not in metadata_cols]
            expr_data = data[expr_cols].select_dtypes(include=[np.number])

            if expr_data.empty:
                raise ValueError("No numeric columns found for scVI synthesis.")

            # Create AnnData object
            adata = anndata.AnnData(X=expr_data.values.astype(np.float32))
            adata.obs_names = [f"cell_{i}" for i in range(len(data))]
            adata.var_names = list(expr_data.columns)

            # Add metadata to obs if present
            if metadata_cols:
                for col in metadata_cols:
                    adata.obs[col] = data[col].values

        # Setup and train scVI model
        n_latent = kwargs.get("n_latent", 10)
        n_layers = kwargs.get("n_layers", 1)
        epochs = kwargs.get("epochs", 100)

        # Work on a copy of adata to avoid modifying the original if passed
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            adata_to_train = adata.copy()
        else:
            adata_to_train = adata

        scvi.model.SCVI.setup_anndata(adata_to_train)
        model = scvi.model.SCVI(
            adata_to_train,
            n_latent=n_latent,
            n_layers=n_layers,
        )

        self.logger.info(f"Training scVI model with {epochs} epochs...")
        model.train(max_epochs=epochs, train_size=0.9, early_stopping=True)

        # Generate synthetic samples by sampling from prior
        self.logger.info(f"Generating {n_samples} synthetic samples...")

        # Sample latent codes from prior (standard normal)
        latent_samples = np.random.randn(n_samples, n_latent).astype(np.float32)

        # Decode latent codes to get synthetic expression
        # We'll use the generative outputs
        import torch

        with torch.no_grad():
            latent_tensor = torch.tensor(latent_samples)
            # Get library size from training data
            # Use adata_to_train to ensure consistency
            library_size = torch.tensor(
                [[adata_to_train.X.sum(axis=1).mean()]] * n_samples
            )

            # Generate from decoder
            generative_outputs = model.module.generative(
                z=latent_tensor,
                library=library_size,
                batch_index=torch.zeros(n_samples, 1, dtype=torch.long),
            )

            # Sample from the distribution
            # In newer scvi-tools, 'px' is a Distribution object (e.g. ZINB)
            # We sample from it to get synthetic counts
            px_dist = generative_outputs["px"]

            # Use sample() to get synthetic counts preserving noise
            # or mean for denoised expression. Sampling is better for synthetic data generation.
            try:
                if hasattr(px_dist, "sample"):
                    synthetic_expression = px_dist.sample()
                elif isinstance(px_dist, torch.Tensor):
                    # Fallback if it returns a tensor directly (e.g. some other likelihoods)
                    synthetic_expression = px_dist
                else:
                    # Last resort, try to get mean
                    synthetic_expression = px_dist.mean
            except Exception:
                # Fallback to mean if sampling fails (e.g. unstable parameters from low training)
                try:
                    synthetic_expression = px_dist.mean
                except Exception:
                    # Absolute fallback if mean also fails: return zeros or latent projection
                    # (This happens if model is completely untrained/unstable)
                    synthetic_expression = torch.zeros(
                        (n_samples, adata.n_vars), dtype=torch.float32
                    )

            if hasattr(synthetic_expression, "cpu"):
                synthetic_expression = synthetic_expression.cpu()

            synth_values = synthetic_expression.numpy()
            print(
                f"DEBUG: scVI synthesis produced values with shape: {synth_values.shape}"
            )

        # Use adata.var_names for column names (works for both DataFrame and AnnData input)
        synth_df = pd.DataFrame(synth_values, columns=adata.var_names)

        # Add metadata column with random sampling from original if present
        if target_col and target_col in (
            data.obs.columns if hasattr(data, "obs") else data.columns
        ):
            # metadata_cols might be empty if adata was passed
            source_metadata = (
                data.obs[target_col].values
                if hasattr(data, "obs")
                else data[target_col].values
            )
            synth_df[target_col] = np.random.choice(
                source_metadata, size=n_samples, replace=True
            )

        self.logger.info(f"scVI synthesis complete. Generated {len(synth_df)} samples.")
        return synth_df

    def _synthesize_gears(
        self,
        data: Union[pd.DataFrame, Any],
        n_samples: int,
        target_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Synthesizes single-cell perturbation data using GEARS (Graph-based perturbation prediction).

        Args:
            data: DataFrame or AnnData with gene expression values
            n_samples: Number of samples to generate
            target_col: Optional column to preserve as metadata
            **kwargs: Additional parameters:
                - perturbations: List of genes to perturb (required)
                - ctrl: Control condition name (default: 'ctrl')
                - epochs: Training epochs (default: 20)
                - batch_size: Batch size (default: 32)
                - device: Device to use (default: 'cpu')

        Returns:
            DataFrame with synthetic perturbation predictions
        """
        self.logger.info(
            "Starting GEARS synthesis for single-cell perturbation data..."
        )

        try:
            from gears import PertData, GEARS
            import anndata
        except ImportError:
            raise ImportError(
                "gears and anndata are required for GEARS synthesis. "
                "Install with: pip install gears anndata"
            )

        # Get perturbations parameter (required)
        perturbations = kwargs.get("perturbations")
        if not perturbations:
            raise ValueError(
                "GEARS requires 'perturbations' parameter: list of genes to perturb. "
                "Example: perturbations=['GENE1', 'GENE2']"
            )

        # Create or use AnnData object
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            adata = data
        else:
            # Separate metadata from expression data
            metadata_cols = []
            if target_col and target_col in data.columns:
                metadata_cols.append(target_col)

            # Get expression columns (numeric only)
            expr_cols = [c for c in data.columns if c not in metadata_cols]
            expr_data = data[expr_cols].select_dtypes(include=[np.number])

            if expr_data.empty:
                raise ValueError("No numeric columns found for GEARS synthesis.")

            # Create AnnData object
            adata = anndata.AnnData(X=expr_data.values.astype(np.float32))
            adata.obs_names = [f"cell_{i}" for i in range(len(data))]
            adata.var_names = list(expr_data.columns)
            adata.var["gene_name"] = list(expr_data.columns)

            # Add condition column if not present
            ctrl_name = kwargs.get("ctrl", "ctrl")
            if "condition" not in adata.obs.columns:
                adata.obs["condition"] = ctrl_name
            adata.obs["cell_type"] = "default"

            if metadata_cols:
                for col in metadata_cols:
                    adata.obs[col] = data[col].values

        # Setup GEARS parameters
        epochs = kwargs.get("epochs", 20)
        batch_size = kwargs.get("batch_size", 32)
        device = kwargs.get("device", "cpu")
        hidden_size = kwargs.get("hidden_size", 64)

        self.logger.info(
            f"Training GEARS model with {epochs} epochs, batch_size={batch_size}..."
        )

        try:
            # Create temporary PertData object
            import tempfile
            import os
            from scipy import sparse

            # GEARS expects sparse matrix for co-expression calculation
            if not sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(adata.X)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Process data for GEARS
                pert_data = PertData(tmpdir)
                pert_data.new_data_process(
                    dataset_name="custom", adata=adata, skip_calc_de=True
                )
                pert_data.load(data_path=os.path.join(tmpdir, "custom"))

                # Inject dummy DE genes metadata if missing (required by GEARS)
                # Must be done AFTER load() because load() reads from disk
                if "non_zeros_gene_idx" not in pert_data.adata.uns:
                    # Create a mapping where each condition maps to all gene indices
                    all_gene_indices = list(range(len(pert_data.adata.var_names)))
                    pert_data.adata.uns["non_zeros_gene_idx"] = {
                        cond: all_gene_indices
                        for cond in pert_data.adata.obs["condition"].unique()
                    }

                # Prepare data split (use all data for training in this case)
                pert_data.prepare_split(split="simulation", seed=1)
                pert_data.get_dataloader(
                    batch_size=batch_size, test_batch_size=batch_size
                )

                # Initialize and train GEARS model
                gears_model = GEARS(pert_data, device=device)
                gears_model.model_initialize(hidden_size=hidden_size)

                try:
                    gears_model.train(epochs=epochs)
                except ValueError as ve:
                    # Ignore pearsonr error during validation on synthetic/small datasets
                    if "at least 2" in str(ve):
                        self.logger.warning(
                            f"GEARS validation metric calculation failed ({ve}). "
                            "Continuing as model training likely completed an epoch."
                        )
                    else:
                        raise ve

                # Generate predictions for specified perturbations
                # Format perturbations as list of lists
                if isinstance(perturbations[0], str):
                    # Single perturbation per prediction
                    pert_list = [[p] for p in perturbations]
                else:
                    # Already formatted as list of lists
                    pert_list = perturbations

                # Predict outcomes
                predictions = gears_model.predict(pert_list)

                # Convert predictions to DataFrame
                if hasattr(predictions, "cpu"):
                    pred_values = predictions.detach().cpu().numpy()
                else:
                    pred_values = predictions

                # Generate n_samples by repeating/sampling predictions
                if len(pred_values) >= n_samples:
                    indices = np.random.choice(
                        len(pred_values), size=n_samples, replace=False
                    )
                else:
                    indices = np.random.choice(
                        len(pred_values), size=n_samples, replace=True
                    )

                synth_values = pred_values[indices]
                synth_df = pd.DataFrame(synth_values, columns=adata.var_names)

                # Add metadata back
                if target_col and target_col in adata.obs.columns:
                    synth_df[target_col] = np.random.choice(
                        adata.obs[target_col], size=n_samples, replace=True
                    )

                self.logger.info(
                    f"GEARS synthesis complete. Generated {len(synth_df)} samples."
                )
                return synth_df

        except Exception as e:
            self.logger.error(f"GEARS synthesis failed: {e}")
            raise e

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

        # If target column is specified and has balanced distribution requested,
        # we balance the STARTING point (bootstrap) so FCS doesn't have to work as hard
        # to move the distribution, and to avoid bias from the starting features.
        X_bootstrap_source = X_real
        if target_col and custom_distributions and target_col in custom_distributions:
            self.logger.info(f"Balancing bootstrap source for column '{target_col}'...")
            X_res, y_res = self._apply_resampling_strategy(
                X_real.drop(columns=target_col),
                X_real[target_col],
                custom_distributions[target_col],
                len(X_real),
            )
            # Reconstruct the full balanced dataframe
            X_bootstrap_source = X_res.copy()
            X_bootstrap_source[target_col] = y_res

        # Ensure object columns are category for consistency
        for col in X_real.select_dtypes(include=["object"]).columns:
            X_real[col] = X_real[col].astype("category")
            X_bootstrap_source[col] = X_bootstrap_source[col].astype("category")

        # Initial random sample
        # OPTIMIZATION: Instead of pure random sample (which might miss rare categories),
        # we repeat the original dataset as many times as possible, then sample the rest.
        n_real = len(X_bootstrap_source)
        if n_samples > n_real:
            n_repeats = n_samples // n_real
            remainder = n_samples % n_real
            X_synth_list = [X_bootstrap_source] * n_repeats
            if remainder > 0:
                X_synth_list.append(
                    X_bootstrap_source.sample(
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
            X_synth = X_bootstrap_source.sample(
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
                            len(Xr_real_train),
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

                    if (
                        is_classification
                        and hasattr(model, "predict_proba")
                        and not (custom_distributions and col in custom_distributions)
                    ):
                        # Probabilistic sampling for better distribution preservation
                        # but we skip it if we are already forcing a distribution via resampling
                        # to avoid double-amplification/overshooting.
                        probs = model.predict_proba(Xs_synth_input)
                        classes = model.classes_

                        # Sample for each row
                        y_synth_pred = np.array(
                            [np.random.choice(classes, p=p) for p in probs]
                        )
                    else:
                        # Regression, balancing via resampling, or fallback
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

    def _apply_resampling_strategy(self, X, y, custom_dist, n_samples):
        """Applies over/under-sampling to match a custom distribution before model training."""
        try:
            original_counts = y.value_counts().to_dict()

            # If "balanced", create a uniform distribution across all present classes
            if custom_dist == "balanced":
                unique_labels = list(original_counts.keys())
                if not unique_labels:
                    return X, y
                prob = 1.0 / len(unique_labels)
                custom_dist = {label: prob for label in unique_labels}

            # Ensure we have a dict
            if not isinstance(custom_dist, dict):
                return X, y

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
        iterations: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with Decision Trees."""

        def model_factory(is_classification):
            try:
                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
            except ImportError:
                raise ImportError("scikit-learn is required for CART synthesis.")

            model_params = {"random_state": self.random_state}
            model_params.update(kwargs)

            # Filter valid params for DecisionTree to avoid the garbage defaults bug
            valid_params = {
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "random_state",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "class_weight",
                "ccp_alpha",
            }
            filtered_params = {
                k: v for k, v in model_params.items() if k in valid_params
            }

            return (
                DecisionTreeClassifier(**filtered_params)
                if is_classification
                else DecisionTreeRegressor(**filtered_params)
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
        iterations: int = 10,
        **kwargs,
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
            model_params.update(kwargs)

            # Filter valid params for RandomForest
            valid_params = {
                "n_estimators",
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "bootstrap",
                "oob_score",
                "n_jobs",
                "random_state",
                "verbose",
                "warm_start",
                "class_weight",
                "ccp_alpha",
                "max_samples",
            }
            filtered_params = {
                k: v for k, v in model_params.items() if k in valid_params
            }

            return (
                RandomForestClassifier(**filtered_params)
                if is_classification
                else RandomForestRegressor(**filtered_params)
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
        iterations: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """Synthesizes data using a Fully Conditional Specification (FCS) approach with LightGBM."""

        def model_factory(is_classification):
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("lightgbm is required for LGBM synthesis.")

            model_params = {"random_state": self.random_state, "verbose": -1}
            model_params.update(kwargs)
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
        data: Union[pd.DataFrame, Any],
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
        drift_injection_config: Optional[List[Union[Dict, DriftConfig]]] = None,
        dynamics_config: Optional[Dict] = None,
        constraints: Optional[List[Dict]] = None,
        adversarial_validation: bool = False,
        report_config: Optional[Union[ReportConfig, Dict]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        The main public method to generate synthetic data.

        Args:
            data (Union[pd.DataFrame, AnnData]): The real dataset (DataFrame) or AnnData object to be synthesized.
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
            **kwargs: Hyperparameters for the models
            adversarial_validation (bool): If True, runs the DiscriminatorReporter to compute adversarial validation metrics (AUC).

        Returns:
            Optional[pd.DataFrame]: The generated synthetic DataFrame, or None if synthesis fails.
        """
        # Handle file paths as input
        if isinstance(data, str):
            import os

            try:
                import anndata
            except ImportError:
                anndata = None

            if not os.path.exists(data):
                self.logger.error(f"File not found: {data}")
                return None

            ext = os.path.splitext(data)[1].lower()
            if ext == ".h5ad":
                try:
                    if anndata is None:
                        raise ImportError("anndata is required for .h5ad files.")
                    self.logger.info(f"Loading AnnData from {data}...")
                    data = anndata.read_h5ad(data)
                except Exception as e:
                    self.logger.error(f"Failed to load .h5ad file: {e}")
                    return None
            elif ext == ".h5":
                try:
                    # Try loading as Pandas HDF5 first
                    self.logger.info(f"Loading HDF5 data from {data}...")
                    # We try common keys or default
                    try:
                        data = pd.read_hdf(data)
                    except (ValueError, KeyError, ImportError):
                        # If it fails, it might be an AnnData stored in H5 format
                        # or requires a specific key
                        if anndata is None:
                            raise ImportError(
                                "anndata is required for AnnData H5 files."
                            )
                        data = anndata.read_h5ad(data)
                except Exception as e:
                    self.logger.error(f"Failed to load .h5 file: {e}")
                    return None
            elif ext == ".csv":
                try:
                    self.logger.info(f"Loading CSV data from {data}...")
                    data = pd.read_csv(data)
                except Exception as e:
                    self.logger.error(f"Failed to load .csv file: {e}")
                    return None
            else:
                self.logger.error(f"Unsupported file format for direct loading: {ext}")
                return None

        # Handle AnnData input
        original_adata = None
        if (
            hasattr(data, "obs")
            and hasattr(data, "X")
            and not isinstance(data, pd.DataFrame)
        ):
            self.logger.info(
                "AnnData input detected. Converting to DataFrame for general processing."
            )
            original_adata = data
            # Convert AnnData to DataFrame for general validation and reporting
            df = data.to_df()
            # Add obs (metadata) to the dataframe
            for col in data.obs.columns:
                df[col] = data.obs[col].values
            data = df

        self._validate_method(method)
        # Note: params was used for default values, now all methods use **kwargs from model_params
        self.logger.info(
            f"Starting generation of {n_samples} samples using method '{method}'..."
        )

        # Resolve ReportConfig (defaults to None if not provided)
        if report_config:
            if isinstance(report_config, dict):
                report_config = ReportConfig(**report_config)
        # We don't necessarily force creation here, reporter handles None.
        # But we might want to consolidate output_dir logic.

        # Determine effective output_dir
        # Logic: 1. report_config.output_dir (if provided/default 'output')?
        #        2. output_dir arg
        #        3. self.output_dir (if exists)
        #        4. '.'

        effective_output_dir = (
            output_dir
            or (report_config.output_dir if report_config else None)
            or getattr(self, "output_dir", None)
            or "."
        )
        # Update report_config if exists
        if report_config:
            report_config.output_dir = effective_output_dir

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
            if method == "ctgan":
                synth = self._synthesize_ctgan(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **kwargs,
                )
            elif method == "tvae":
                synth = self._synthesize_tvae(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **kwargs,
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
                    **(kwargs or {}),
                )
            elif method == "copula":
                synth = self._synthesize_copula(
                    data,
                    n_samples,
                    "copula",
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "rf":
                synth = self._synthesize_rf(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "lgbm":
                synth = self._synthesize_lgbm(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "gmm":
                synth = self._synthesize_gmm(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )

            elif method == "smote":
                synth = self._synthesize_smote(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )
            elif method == "adasyn":
                synth = self._synthesize_adasyn(
                    data,
                    n_samples,
                    target_col=target_col,
                    custom_distributions=custom_distributions,
                    **(kwargs or {}),
                )

            elif method == "diffusion":
                synth = self._synthesize_diffusion(data, n_samples, **(kwargs or {}))
            elif method == "ddpm":
                synth = self._synthesize_ddpm(data, n_samples, **(kwargs or {}))
            elif method == "timegan":
                synth = self._synthesize_timegan(data, n_samples, **(kwargs or {}))
            elif method == "timevae":
                synth = self._synthesize_timevae(data, n_samples, **(kwargs or {}))
            elif method == "scvi":
                # Pass original_adata if available to avoid redundant conversion
                synth = self._synthesize_scvi(
                    original_adata if original_adata is not None else data,
                    n_samples,
                    target_col=target_col,
                    **(kwargs or {}),
                )
            elif method == "gears":
                # Pass original_adata if available to avoid redundant conversion
                synth = self._synthesize_gears(
                    original_adata if original_adata is not None else data,
                    n_samples,
                    target_col=target_col,
                    **(kwargs or {}),
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
                if synth is not None and dynamics_config:
                    print("DEBUG: Applying dynamics config...")
                    self.logger.info("Applying dynamics injection config...")
                    from calm_data_generator.generators.dynamics.ScenarioInjector import (
                        ScenarioInjector,
                    )

                    injector = ScenarioInjector(
                        random_state=self.random_state, logger=self.logger
                    )
                    synth = injector.apply_config(synth, dynamics_config)

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
                if synth is not None and drift_injection_config:
                    print("DEBUG: Applying drift injection...")
                    self.logger.info("Applying drift injection config...")
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
                        # Determine method and params
                        method_name = "inject_feature_drift"  # Default
                        params_drift = {}
                        drift_obj = None

                        if isinstance(drift_conf, DriftConfig):
                            method_name = drift_conf.method
                            drift_obj = drift_conf
                            params_drift = drift_conf.params or {}
                        elif isinstance(drift_conf, dict):
                            # Support nested {"method": ..., "params": ...} or flat
                            if "method" in drift_conf and "params" in drift_conf:
                                method_name = drift_conf.get("method")
                                params_drift = drift_conf.get("params", {})
                            else:
                                # Flat dict
                                method_name = drift_conf.get(
                                    "drift_method",
                                    drift_conf.get("method", "inject_feature_drift"),
                                )
                                params_drift = drift_conf

                        if hasattr(drift_injector, method_name):
                            self.logger.info(f"Injecting drift: {method_name}")
                            drift_method = getattr(drift_injector, method_name)
                            try:
                                # Add 'df' if not present
                                if "df" not in params_drift:
                                    params_drift["df"] = synth

                                # Call method
                                if drift_obj:
                                    # Pass config object explicitly
                                    res = drift_method(
                                        drift_config=drift_obj, **params_drift
                                    )
                                else:
                                    # Pass params (will be converted to config internally if needed)
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
                    print("DEBUG: Generating report...")
                    time_col_name = date_config.date_col if date_config else "timestamp"

                    # Build drift_config for report if drift was applied
                    report_drift_config = None
                    if drift_injection_config:
                        # Summarize drift configuration for the report
                        drift_methods = []
                        for d in drift_injection_config:
                            if isinstance(d, DriftConfig):
                                drift_methods.append(d.method)
                            else:
                                drift_methods.append(
                                    d.get("method", d.get("drift_method", "unknown"))
                                )

                        report_drift_config = {
                            "drift_type": ", ".join(drift_methods),
                            "drift_magnitude": "See config",
                            "affected_columns": "Multiple (via drift_injection_config)",
                        }

                    self.reporter.generate_comprehensive_report(
                        real_df=data,
                        synthetic_df=synth,
                        generator_name=f"RealGenerator_{method}",
                        output_dir=effective_output_dir
                        or output_dir,  # Use effective dir
                        target_column=target_col,
                        time_col=time_col_name,
                        drift_config=report_drift_config,
                        adversarial_validation=adversarial_validation,
                        report_config=report_config,  # Pass the config object
                    )

                # Save the generated dataset for inspection
                if save_dataset:  # Only save if save_dataset is True
                    print("DEBUG: Saving dataset...")
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

                print(f"DEBUG: Returning synth for method '{method}'.")
                return synth
            else:
                print(
                    f"DEBUG: Synthesis method '{method}' failed to generate data (synth is None)."
                )
                self.logger.error(
                    f"Synthesis method '{method}' failed to generate data."
                )
                return None
        except Exception as e:
            print(f"DEBUG: Synthesis with method '{method}' failed with exception: {e}")
            self.logger.error(
                f"Synthesis with method '{method}' failed: {e}", exc_info=True
            )
            return None
