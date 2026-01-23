#!/usr/bin/env python3
"""
Drift Injector for Real Data - Injects various types of drift into real datasets.

This module provides the DriftInjector class, which is designed to introduce a wide range of controlled
drifts into a pandas DataFrame. It supports various drift types, including feature drift, label drift,
and more complex patterns like gradual, abrupt, and recurrent drifts.

Key Features:
- **Multiple Drift Types**: Inject gaussian_noise, shift, scale, and other transformations.
- **Flexible Targeting**: Apply drift to the entire dataset, specific blocks, or row indices.
- **Advanced Drift Profiles**: Simulate gradual, abrupt, incremental, and recurrent drifts using window functions (sigmoid, linear, cosine).
- **Label and Concept Drift**: Includes methods for label flipping (label_drift), changing target distribution (label_shift), and introducing new categories (new_category_drift).
- **Covariate and Virtual Drift**: Modify correlation structures (correlation_matrix_drift) and introduce missing values (missing_values_drift).
- **Integrated Reporting**: Automatically generates detailed reports and visualizations comparing the original and drifted datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Sequence, Tuple, Any, Union
import warnings
import os
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DriftInjector:
    """
    A class to inject various types of drift into a pandas DataFrame.
    """

    # -------------------------
    # Init & utils
    # -------------------------
    def __init__(
        self,
        output_dir: str = "drift_output",
        generator_name: str = "DriftInjector",
        random_state: Optional[int] = None,
        time_col: Optional[str] = None,
        block_column: Optional[str] = None,
        target_column: Optional[str] = None,
        original_df: Optional[pd.DataFrame] = None,
        minimal_report: bool = False,
        auto_report: bool = True,
    ):
        """
        Initializes the DriftInjector.

        Args:
            output_dir (str): Default directory to save reports and drifted datasets.
            generator_name (str): Default name for the generator, used in output file names.
            random_state (Optional[int]): Seed for the random number generator for reproducibility.
            auto_report (bool): If True, automatically generates reports after drift injection.
            minimal_report (bool): If True, generates minimal reports (faster, no correlations/PCA).
        """
        self.rng = np.random.default_rng(random_state)
        self.output_dir = output_dir
        self.generator_name = generator_name
        self.random_state = random_state
        self.time_col = time_col
        self.block_column = block_column
        self.target_column = target_column
        self.auto_report = auto_report
        self.minimal_report = minimal_report

        self.reporter = QualityReporter(minimal=minimal_report)

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def _frac(self, x: float) -> float:
        """Clips a float to the [0.0, 1.0] range."""
        return float(np.clip(x, 0.0, 1.0))

    def _generate_reports(
        self,
        original_df,
        drifted_df,
        drift_config,
        time_col: Optional[str] = None,
        resample_rule: Optional[Union[str, int]] = None,
    ):
        """Helper to generate the standard report."""
        # Generate the primary report in the main output directory
        self.reporter.update_report_after_drift(
            original_df=original_df,
            drifted_df=drifted_df,
            output_dir=self.output_dir,
            drift_config=drift_config,
            time_col=time_col,
            resample_rule=resample_rule,
        )

    def _ensure_psd_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Ensures a matrix is positive semi-definite (PSD) by adjusting its eigenvalues."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues[eigenvalues < 1e-6] = 1e-6  # Clamp small eigenvalues
        psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Renormalize to have 1s on the diagonal
        d = np.sqrt(np.diag(psd_matrix))
        d_inv = np.where(d > 1e-9, 1.0 / d, 0)
        psd_matrix = np.diag(d_inv) @ psd_matrix @ np.diag(d_inv)
        return psd_matrix

    def _get_target_rows(
        self,
        df: pd.DataFrame,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        index_step: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        block_step: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        time_step: Optional[Any] = None,
        **kwargs,
    ) -> pd.Index:
        """
        Selects rows for drift injection based on a hierarchy of criteria.
        """
        # Fallback to instance attributes if not provided
        block_column = block_column or self.block_column
        time_col = time_col or self.time_col

        if time_start or time_end or time_ranges or specific_times:
            return self._select_rows_by_time(
                df,
                time_col=time_col,
                time_start=time_start,
                time_end=time_end,
                time_ranges=time_ranges,
                specific_times=specific_times,
                time_step=time_step,
            )
        if blocks is not None or block_start is not None:
            return self._select_rows_by_blocks(
                df,
                block_column=block_column,
                blocks=blocks,
                block_start=block_start,
                n_blocks=n_blocks,
                block_step=block_step,
            )
        if block_index is not None:
            used_block_column = block_column
            if used_block_column not in df.columns:
                raise ValueError(f"Block column '{used_block_column}' not found")
            return df.index[df[used_block_column] == block_index]
        if start_index is not None or end_index is not None:
            return self._select_rows_by_index(df, start_index, end_index, index_step)

        return df.index

    def _select_rows_by_index(
        self,
        df: pd.DataFrame,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: Optional[int] = None,
    ) -> pd.Index:
        """
        Selects rows by index range and step.
        """
        start = start if start is not None else 0
        end = end if end is not None else len(df)
        step = step if step is not None else 1
        return df.iloc[start:end:step].index

    # -------------------------
    # Advanced time selection
    # -------------------------
    def _select_rows_by_time(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        time_step: Optional[Any] = None,
    ) -> pd.Index:
        """
        Selects rows based on time criteria.
        """
        if not time_col or time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' is required")

        time_series = pd.to_datetime(df[time_col])
        mask = pd.Series(False, index=df.index)

        if specific_times:
            mask |= time_series.isin(pd.to_datetime(specific_times))

        if time_ranges:
            for start, end in time_ranges:
                mask |= (time_series >= pd.to_datetime(start)) & (
                    time_series <= pd.to_datetime(end)
                )

        if time_start or time_end:
            start_dt = pd.to_datetime(time_start) if time_start else pd.Timestamp.min
            end_dt = pd.to_datetime(time_end) if time_end else pd.Timestamp.max
            mask |= (time_series >= start_dt) & (time_series <= end_dt)

        if time_step:
            if not (time_start and time_end):
                raise ValueError(
                    "time_start and time_end are required when using time_step"
                )
            step_range = pd.date_range(start=time_start, end=time_end, freq=time_step)
            mask &= time_series.isin(step_range)

        return df.index[mask]

    # -------------------------
    # Advanced block selection
    # -------------------------
    def _select_rows_by_blocks(
        self,
        df: pd.DataFrame,
        block_column: str,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        block_step: Optional[int] = None,
    ) -> pd.Index:
        """
        Selects rows based on block identifiers.
        """
        if not block_column or block_column not in df.columns:
            raise ValueError(f"Block column '{block_column}' is required")

        if blocks:
            return df.index[df[block_column].isin(blocks)]

        if block_start is not None:
            uniq = sorted(df[block_column].dropna().unique())
            if block_start not in uniq:
                warnings.warn(f"block_start '{block_start}' not in '{block_column}'")
                return df.iloc[0:0].index

            i0 = uniq.index(block_start)
            n_blocks = n_blocks if n_blocks is not None else len(uniq) - i0
            block_step = block_step if block_step is not None else 1

            selected_blocks = uniq[i0 : i0 + n_blocks * block_step : block_step]
            return df.index[df[block_column].isin(selected_blocks)]

        return df.iloc[0:0].index

    # -------------------------
    # Windows (profiles + speed)
    # -------------------------
    def _sigmoid_weights(self, n: int, center: float, width: int) -> np.ndarray:
        """
        Creates weights w in [0,1] over n positions with a sigmoid transition.

        Args:
            n (int): Number of positions (rows).
            center (float): The center of the transition (in coordinates 0..n-1).
            width (int): Controls how many rows it takes to go from ~10% to ~90%.

        Returns:
            np.ndarray: An array of weights.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)
        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        # Map width -> sigmoid scale. Approximately 4*scale ~ width (10%->90%)
        scale = width / 4.0
        z = (i - float(center)) / max(1e-9, scale)
        w = 1.0 / (1.0 + np.exp(-z))
        return w

    def _window_weights(
        self,
        n: int,
        center: float,
        width: int,
        profile: str = "sigmoid",
        k: float = 1.0,
        direction: str = "up",
    ) -> np.ndarray:
        """
        Returns weights w in [0,1] of size n with a transition centered at `center` and of `width`.

        Args:
            n (int): Number of positions.
            center (float): Center of the transition.
            width (int): Width of the transition.
            profile (str): The shape of the transition window ("sigmoid", "linear", "cosine").
            k (float): Controls the "speed" (slope) of the transition.
            direction (str): "up" (0->1) or "down" (1->0).

        Returns:
            np.ndarray: An array of weights.
        """
        if n <= 0:
            return np.zeros(0, dtype=float)

        i = np.arange(n, dtype=float)
        width = max(1, int(width))
        center = float(center)

        if profile == "sigmoid":
            base_scale = width / 4.0
            scale = max(1e-9, base_scale / max(1e-9, float(k)))  # high k -> faster
            z = (i - center) / scale
            w = 1.0 / (1.0 + np.exp(-z))
        elif profile == "linear":
            left = center - width / 2.0
            right = center + width / 2.0
            w = (i - left) / max(1e-9, (right - left))
            w = np.clip(w, 0.0, 1.0)
            if k != 1.0:
                w = np.clip((w - 0.5) * k + 0.5, 0.0, 1.0)
        elif profile == "cosine":
            left = center - width / 2.0
            right = center + width / 2.0
            t = (i - left) / max(1e-9, (right - left))
            t = np.clip(t, 0.0, 1.0)
            w = 0.5 - 0.5 * np.cos(np.pi * t)
            if k != 1.0:
                w = np.clip((w - 0.5) * k + 0.5, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown profile: {profile}")

        if direction == "down":
            w = 1.0 - w

        return w

    def _calculate_drift_probabilities(
        self,
        rows: pd.Index,
        center: int,
        width: int,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        index_min: Optional[int] = None,
        index_max: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calculates drift probabilities for a given set of rows based on window parameters.

        Args:
            rows: The row indices to calculate probabilities for.
            center: Center of the transition window.
            width: Width of the transition window.
            profile: Window profile ("sigmoid", "linear", "cosine").
            speed_k: Speed factor for the transition.
            direction: "up" (0->1) or "down" (1->0).
            index_min: Minimum index for normalization (default: min(rows)).
            index_max: Maximum index for normalization (default: max(rows)).

        Returns:
            np.ndarray: Array of probabilities for each row.
        """
        n = len(rows)
        if n == 0:
            return np.zeros(0, dtype=float)

        # Validate row indices
        if len(rows) == 0:
            return np.zeros(0, dtype=float)

        # Compute window weights based on positions
        return self._window_weights(
            n,
            center=float(center),
            width=int(width),
            profile=profile,
            k=speed_k,
            direction=direction,
        )

    # -------------------------
    # Common engine for features
    # -------------------------
    def _apply_numeric_op_with_weights(
        self,
        values: np.ndarray,
        drift_type: str,
        drift_magnitude: float,
        w: np.ndarray,
        rng: np.random.Generator,
        column_drift_value: Optional[float],
    ) -> np.ndarray:
        """
        Applies a numeric drift operation, scaled by weights `w` row by row.
        """
        x = values.astype(float, copy=True)
        n = len(x)
        if n == 0:
            return x

        mean = float(np.mean(x)) if n > 0 else 0.0
        std = float(np.std(x)) if n > 0 else 0.0
        w = np.asarray(w, dtype=float)

        # Fix for broadcasting error when w is shorter than x
        if len(w) < n:
            w = np.pad(w, (0, n - len(w)), "edge")

        w = np.clip(w, 0.0, 1.0)

        if drift_type == "gaussian_noise":
            if std == 0:
                return x
            noise = rng.normal(0.0, drift_magnitude * std, size=n)
            x = x + noise * w

        elif drift_type == "shift":
            shift_amt = drift_magnitude * mean
            x = x + shift_amt * w

        elif drift_type == "scale":
            # row-wise factor: 1 + w*m
            factor = 1.0 + (w * drift_magnitude)
            x = mean + (x - mean) * factor

        elif drift_type == "add_value":
            if column_drift_value is None:
                raise ValueError("add_value requires drift_value/drift_values[col]")
            x = x + (w * column_drift_value)

        elif drift_type == "subtract_value":
            if column_drift_value is None:
                raise ValueError(
                    "subtract_value requires drift_value/drift_values[col]"
                )
            x = x - (w * column_drift_value)

        elif drift_type == "multiply_value":
            if column_drift_value is None:
                raise ValueError(
                    "multiply_value requires drift_value/drift_values[col]"
                )
            # mix towards the indicated factor: x * (1 + w*(f-1))
            factor = 1.0 + w * (float(column_drift_value) - 1.0)
            x = x * factor

        elif drift_type == "divide_value":
            if column_drift_value is None:
                raise ValueError("divide_value requires drift_value/drift_values[col]")
            if float(column_drift_value) == 0.0:
                raise ValueError("drift_value cannot be zero for 'divide_value'")
            # dividing is equivalent to multiplying by (1/val); we mix towards that factor
            target = 1.0 / float(column_drift_value)
            factor = 1.0 + w * (target - 1.0)
            x = x * factor

        elif drift_type == "uniform_noise":
            # Uniform noise in range [-magnitude*std, +magnitude*std]
            if std == 0:
                return x
            noise = rng.uniform(-drift_magnitude * std, drift_magnitude * std, size=n)
            x = x + noise * w

        else:
            raise ValueError(f"Unknown drift_type: {drift_type}")

        # Preserve original dtype to avoid FutureWarnings
        original_dtype = values.dtype
        if pd.api.types.is_integer_dtype(original_dtype):
            x = np.round(x).astype(original_dtype)

        return x

    def _apply_categorical_with_weights(
        self,
        original_vals: pd.Series,
        w: np.ndarray,
        drift_magnitude: float,
        rng: np.random.Generator,
    ) -> pd.Series:
        """
        Changes categorical values with a probability per row p = clamp(w * drift_magnitude).
        Replaces the value with a random category different from the current one.
        """
        s = original_vals.copy()
        uniques = s.dropna().unique()
        if len(uniques) <= 1:
            return s

        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        p = np.clip(w * self._frac(drift_magnitude), 0.0, 1.0)

        # flip a coin for each row
        mask = rng.random(len(s)) < p
        idxs = s.index[mask]
        if len(idxs) == 0:
            return s

        # for each row to change, choose a different category
        current = s.loc[idxs].to_numpy()
        new_vals = []
        for cur in current:
            choices = [u for u in uniques if u != cur]
            if choices:
                new_vals.append(rng.choice(choices))
            else:
                new_vals.append(cur)
        s.loc[idxs] = new_vals
        return s

    def _validate_feature_op(self, drift_type: str, drift_magnitude: float):
        """Validates the feature drift operation and its magnitude."""
        if drift_type in {"gaussian_noise", "shift", "scale"} and drift_magnitude < 0:
            raise ValueError(
                "drift_magnitude must be >= 0 for gaussian_noise/shift/scale"
            )
        valid = {
            "gaussian_noise",
            "shift",
            "scale",
            "add_value",
            "subtract_value",
            "multiply_value",
            "divide_value",
            "uniform_noise",
        }
        if drift_type not in valid:
            raise ValueError(f"Unknown drift_type: {drift_type}")

    def inject_feature_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Applies drift at once based on various selection criteria.

        Args:
            df, feature_cols, drift_type, drift_magnitude, drift_value, drift_values: Core drift parameters.
            start_index, block_index, block_column: Index and block-based selection.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            time_col: The timestamp column used for time selection.
            auto_report: Whether to generate a report.
            output_dir: Directory for reports (overrides init default).
            generator_name: Name for reports (overrides init default).
        """
        self._validate_feature_op(drift_type, drift_magnitude)
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            **kwargs,
        )

        w = np.ones(len(rows), dtype=float)

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found")
                continue

            column_drift_value = None
            if drift_type in {
                "add_value",
                "subtract_value",
                "multiply_value",
                "divide_value",
            }:
                column_drift_value = (
                    drift_values.get(col) if drift_values else drift_value
                )
                if column_drift_value is None:
                    raise ValueError(
                        f"For '{drift_type}', provide drift_value or drift_values['{col}']"
                    )

            if pd.api.types.is_numeric_dtype(df[col]):
                x = df_drift.loc[rows, col].to_numpy(copy=True)
                x2 = self._apply_numeric_op_with_weights(
                    x, drift_type, drift_magnitude, w, self.rng, column_drift_value
                )
                df_drift.loc[rows, col] = x2
            else:
                s = df_drift.loc[rows, col]
                s2 = self._apply_categorical_with_weights(
                    s, w, drift_magnitude, self.rng
                )
                df_drift.loc[rows, col] = s2

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir

            drift_config = {
                "drift_method": "inject_feature_drift",
                "feature_cols": feature_cols,
                "drift_type": drift_type,
                "drift_magnitude": drift_magnitude,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_feature_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(
                    df,
                    df_drift,
                    drift_config,
                    time_col=time_col,
                    resample_rule=kwargs.get("resample_rule"),
                )
        return df_drift

    # -------------------------
    # Feature drift “windowed”: gradual, abrupt, incremental, recurrent
    # -------------------------
    def inject_feature_drift_gradual(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        inconsistency: float = 0.0,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        resample_rule: Optional[Union[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Injects gradual drift on selected rows using a transition window.
        """
        self._validate_feature_op(drift_type, drift_magnitude)
        df_drift = df.copy()

        rows = self._get_target_rows(
            df,
            start_index=start_index,
            end_index=end_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))
        w = self._window_weights(
            n,
            center=c,
            width=w_width,
            profile=profile,
            k=float(speed_k),
            direction=direction,
        )

        if inconsistency > 0:
            # Simplified inconsistency logic for brevity
            noise = self.rng.normal(0, 0.1 * inconsistency, n)
            w = np.clip(w + noise, 0.0, 1.0)

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found")
                continue

            column_drift_value = drift_values.get(col) if drift_values else drift_value
            if pd.api.types.is_numeric_dtype(df[col]):
                x = df_drift.loc[rows, col].to_numpy(copy=True)
                x2 = self._apply_numeric_op_with_weights(
                    x, drift_type, drift_magnitude, w, self.rng, column_drift_value
                )
                df_drift.loc[rows, col] = x2
            else:
                s = df_drift.loc[rows, col]
                s2 = self._apply_categorical_with_weights(
                    s, w, drift_magnitude, self.rng
                )
                df_drift.loc[rows, col] = s2

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_feature_drift_gradual",
                "feature_cols": feature_cols,
                "drift_type": drift_type,
                "drift_magnitude": drift_magnitude,
                "profile": profile,
                "center": center,
                "width": width,
                "inconsistency": inconsistency,
                "generator_name": f"{gen_name}_feature_gradual",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(
                    df,
                    df_drift,
                    drift_config,
                    time_col=time_col,
                    resample_rule=resample_rule,
                )
        return df_drift

    def inject_feature_drift_incremental(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects a constant and smooth drift using a single wide sigmoid transition.
        """
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )
        n = len(rows)
        if n == 0:
            return df.copy()

        center = n / 2
        width = n

        return self.inject_feature_drift_gradual(
            df=df,
            feature_cols=feature_cols,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            center=int(round(center)),
            width=width,
            profile="sigmoid",
            speed_k=1.0,
            direction="up",
            **kwargs,
        )

    def inject_feature_drift_recurrent(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str,
        drift_magnitude: float,
        drift_value: Optional[float] = None,
        drift_values: Optional[Dict[str, float]] = None,
        windows: Optional[Sequence[Tuple[int, int]]] = None,
        block_column: Optional[str] = None,
        cycle_blocks: Optional[Sequence] = None,
        repeats: int = 1,
        random_repeat_order: bool = False,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects recurrent drift by applying several drift windows.
        """
        df_out = df.copy()

        # This method's logic gets complex with time selection.
        # For now, we assume 'windows' applies to the selected rows from time/block criteria.
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        # The logic for 'cycle_blocks' and 'windows' needs careful integration with the new time selection.
        # This is a simplified version.
        if not rows.empty:
            # Apply drift to the selected rows
            n_total = len(rows)
            chunk_size = n_total // max(1, repeats)

            for i in range(repeats):
                # Determine start/end of this recurrence window
                c_start = i * chunk_size
                c_end = (i + 1) * chunk_size if i < repeats - 1 else n_total

                if c_end <= c_start:
                    continue

                chunk_indices = rows[c_start:c_end]
                n_chunk = len(chunk_indices)

                # Determine local center/width for this chunk
                local_center = n_chunk // 2 if center is None else center
                local_width = n_chunk if width is None else width

                w = self._window_weights(
                    n_chunk,
                    float(local_center),
                    int(local_width),
                    profile,
                    speed_k,
                    direction,
                )

                # Apply to each feature
                for col in feature_cols:
                    if col not in df_out.columns:
                        continue

                    column_drift_value = (
                        drift_values.get(col) if drift_values else drift_value
                    )

                    if pd.api.types.is_numeric_dtype(df_out[col]):
                        x = df_out.loc[chunk_indices, col].to_numpy(copy=True)
                        x2 = self._apply_numeric_op_with_weights(
                            x,
                            drift_type,
                            drift_magnitude,
                            w,
                            self.rng,
                            column_drift_value,
                        )
                        df_out.loc[chunk_indices, col] = x2
                    else:
                        s = df_out.loc[chunk_indices, col]
                        s2 = self._apply_categorical_with_weights(
                            s, w, drift_magnitude, self.rng
                        )
                        df_out.loc[chunk_indices, col] = s2

        return df_out

    def inject_conditional_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        conditions: List[Dict[str, Any]],
        drift_type: str,
        drift_magnitude: float,
        drift_method: str = "abrupt",  # New parameter: abrupt, gradual, incremental
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        index_step: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        block_step: Optional[int] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        time_step: Optional[Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects drift (abrupt, gradual, etc.) on a data subset defined by conditions.

        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_cols (List[str]): Columns to apply drift to.
            conditions (List[Dict[str, Any]]): A list of dicts defining filters (e.g. [{"column": "age", "operator": ">", "value": 50}]).
            drift_type (str): Type of numeric/categorical drift.
            drift_magnitude (float): Magnitude of the drift.
            drift_method (str): Method of injection: "abrupt" (default), "gradual", "incremental".
            **kwargs: Additional args passed to the underlying injection method (e.g. center, width, profile for gradual).
        """
        df_drift = df.copy()

        # 1. Select initial candidate rows based on index/block/time
        base_rows = self._get_target_rows(
            df,
            start_index=start_index,
            end_index=end_index,
            index_step=index_step,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            block_step=block_step,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            time_step=time_step,
        )

        # 2. Apply filtering conditions
        final_mask = pd.Series(True, index=base_rows)
        for condition in conditions:
            col = condition["column"]
            op = condition["operator"]
            val = condition["value"]

            if col not in df.columns:
                raise ValueError(f"Condition column '{col}' not found in dataframe")

            # Safe access to column series
            series = df.loc[base_rows, col]

            if op == ">":
                final_mask &= series > val
            elif op == ">=":
                final_mask &= series >= val
            elif op == "<":
                final_mask &= series < val
            elif op == "<=":
                final_mask &= series <= val
            elif op == "==":
                final_mask &= series == val
            elif op == "!=":
                final_mask &= series != val
            elif op == "in":
                final_mask &= series.isin(val)
            else:
                raise ValueError(f"Unsupported operator: {op}")

        target_rows_idx = base_rows[final_mask]

        if target_rows_idx.empty:
            warnings.warn("No rows matched the conditions. No drift injected.")
            return df

        # 3. Dynamic Dispatch based on drift_method
        subset_df = df.loc[target_rows_idx].copy()

        # Prepare common kwargs
        method_kwargs = {
            "feature_cols": feature_cols,
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            **kwargs,
        }

        if drift_method == "abrupt":
            # Uses standard feature drift (supports auto_report)
            method_kwargs["auto_report"] = False
            drifted_subset = self.inject_feature_drift(df=subset_df, **method_kwargs)

        elif drift_method == "gradual":
            # Ensure center/width relate to the subset size if not provided
            if "width" not in kwargs:
                method_kwargs["width"] = len(subset_df)

            drifted_subset = self.inject_feature_drift_gradual(
                df=subset_df, **method_kwargs
            )

        elif drift_method == "incremental":
            drifted_subset = self.inject_feature_drift_incremental(
                df=subset_df, **method_kwargs
            )

        elif drift_method == "recurrent":
            # Recurrent requires 'repeats' parameter
            if "repeats" not in method_kwargs:
                method_kwargs["repeats"] = 2  # Default to 2 cycles
            drifted_subset = self.inject_feature_drift_recurrent(
                df=subset_df, **method_kwargs
            )

        else:
            raise ValueError(
                f"Unknown drift_method: {drift_method}. Use 'abrupt', 'gradual', 'incremental', or 'recurrent'."
            )

        # 4. Update and Report
        df_drift.update(drifted_subset)

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_conditional_drift",
                "sub_method": drift_method,
                "feature_cols": feature_cols,
                "conditions": conditions,
                "drift_type": drift_type,
                "drift_magnitude": drift_magnitude,
                "generator_name": f"{self.generator_name}_conditional_{drift_method}",
                **kwargs,
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    # -------------------------
    # Label drift
    # -------------------------
    def inject_label_drift(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        drift_magnitude: float = 0.1,
        drift_magnitudes: Optional[Dict[str, float]] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects random label flips for a specified section.

        Args:
            df: Input DataFrame.
            target_cols: List of target columns to apply label drift to.
            drift_magnitude: Fraction of labels to flip (0.0 to 1.0).
            drift_magnitudes: Per-column magnitudes (overrides drift_magnitude).
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            auto_report: Whether to generate a report.
            output_dir: Directory for reports.
            generator_name: Name for reports.

        Returns:
            pd.DataFrame: DataFrame with label drift applied.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        if len(rows) == 0:
            return df_drift

        for col in target_cols:
            if col not in df.columns:
                warnings.warn(f"Target column '{col}' not found")
                continue

            col_magnitude = (
                drift_magnitudes.get(col, drift_magnitude)
                if drift_magnitudes
                else drift_magnitude
            )
            col_magnitude = self._frac(col_magnitude)

            unique_labels = df[col].dropna().unique()
            if len(unique_labels) <= 1:
                warnings.warn(f"Column '{col}' has only one unique value, skipping")
                continue

            # Determine number of rows to flip
            n_to_flip = int(len(rows) * col_magnitude)
            if n_to_flip == 0:
                continue

            # Select random rows to flip
            flip_indices = self.rng.choice(rows, size=n_to_flip, replace=False)

            # Flip labels
            for idx in flip_indices:
                current_val = df_drift.at[idx, col]
                # Choose a different label
                possible_vals = [v for v in unique_labels if v != current_val]
                if possible_vals:
                    df_drift.at[idx, col] = self.rng.choice(possible_vals)

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_label_drift",
                "target_cols": target_cols,
                "drift_magnitude": drift_magnitude,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_label_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    def inject_label_drift_gradual(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float = 0.3,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        inconsistency: float = 0.0,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects gradual label drift using a transition window.

        The probability of flipping a label increases gradually based on the
        window profile (sigmoid, linear, cosine).

        Args:
            df: Input DataFrame.
            target_col: Target column to apply drift to.
            drift_magnitude: Maximum fraction of labels to flip (0.0 to 1.0).
            start_index, end_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            center: Center of the transition window (default: middle of selected rows).
            width: Width of the transition window (default: n/5).
            profile: Window profile ("sigmoid", "linear", "cosine").
            speed_k: Speed of transition (higher = faster).
            direction: "up" (increasing flip probability) or "down" (decreasing).
            inconsistency: Random noise to add to probabilities.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with gradual label drift applied.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            end_index=end_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        unique_labels = df[target_col].dropna().unique()
        if len(unique_labels) <= 1:
            warnings.warn(f"Column '{target_col}' has only one unique value")
            return df_drift

        # Calculate transition weights
        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))
        w = self._window_weights(
            n,
            center=c,
            width=w_width,
            profile=profile,
            k=float(speed_k),
            direction=direction,
        )

        # Add inconsistency noise
        if inconsistency > 0:
            noise = self.rng.normal(0, 0.1 * inconsistency, n)
            w = np.clip(w + noise, 0.0, 1.0)

        # Calculate flip probability for each row
        flip_probs = w * self._frac(drift_magnitude)

        # Apply gradual flips
        for i, idx in enumerate(rows):
            if self.rng.random() < flip_probs[i]:
                current_val = df_drift.at[idx, target_col]
                possible_vals = [v for v in unique_labels if v != current_val]
                if possible_vals:
                    df_drift.at[idx, target_col] = self.rng.choice(possible_vals)

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_label_drift_gradual",
                "target_col": target_col,
                "drift_magnitude": drift_magnitude,
                "profile": profile,
                "center": center,
                "width": width,
                "generator_name": f"{gen_name}_label_gradual",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    def inject_label_drift_abrupt(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float,
        change_index: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Wrapper for a very fast gradual drift to simulate an abrupt change."""
        return self.inject_label_drift_gradual(
            df=df,
            target_col=target_col,
            drift_magnitude=drift_magnitude,
            center=change_index,
            width=3,
            speed_k=5.0,
            **kwargs,
        )

    def inject_label_drift_incremental(
        self, df: pd.DataFrame, target_col: str, drift_magnitude: float, **kwargs
    ) -> pd.DataFrame:
        """Applies a constant and smooth label drift over the selected rows."""
        # Extract auto_report before passing kwargs to avoid duplicate
        auto_report = kwargs.pop("auto_report", True)

        rows = self._get_target_rows(df, **kwargs)
        n = len(rows)
        if n == 0:
            return df.copy()

        center = n / 2
        width = n

        return self.inject_label_drift_gradual(
            df=df,
            target_col=target_col,
            drift_magnitude=drift_magnitude,
            center=int(round(center)),
            width=width,
            auto_report=auto_report,
            **kwargs,
        )

    def inject_label_drift_recurrent(
        self,
        df: pd.DataFrame,
        target_col: str,
        drift_magnitude: float,
        windows: List[Tuple[int, int]],
        **kwargs,
    ) -> pd.DataFrame:
        """Applies label drift over a series of explicit windows."""
        df_out = df.copy()
        for center, width in windows:
            df_out = self.inject_label_drift_gradual(
                df=df_out,
                target_col=target_col,
                drift_magnitude=drift_magnitude,
                center=center,
                width=width,
                auto_report=False,
                **kwargs,
            )
        # Final reporting logic
        return df_out

    # -------------------------
    # Target distribution drift
    # -------------------------
    def inject_outliers_global(
        self,
        df: pd.DataFrame,
        cols: List[str],
        outlier_prob: float = 0.05,
        factor: float = 3.0,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects global outliers by scaling values by a factor.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            time_col=time_col,
        )

        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Randomly select rows to outlier
                mask = self.rng.random(len(rows)) < outlier_prob
                outlier_rows = rows[mask]

                # Apply factor (random sign)
                signs = self.rng.choice([-1, 1], size=len(outlier_rows))
                df_drift.loc[outlier_rows, col] += (
                    factor * df_drift.loc[outlier_rows, col].std() * signs
                )
            else:
                warnings.warn(f"Global outliers failed for column {col}.")

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_outliers_global",
                "cols": cols,
                "outlier_prob": outlier_prob,
                "factor": factor,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_global_outliers",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )

        return df_drift

    # -------------------------
    def inject_new_value(
        self,
        df: pd.DataFrame,
        cols: List[str],
        new_value: Any,  # Or a distribution function
        prob: float = 1.0,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects a completely new value into a column (Categorical shift).
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            time_col=time_col,
            **kwargs,
        )

        for col in cols:
            if col in df.columns:
                mask = self.rng.random(len(rows)) < prob
                rows_mod = rows[mask]
                df_drift.loc[rows_mod, col] = new_value
            else:
                warnings.warn(f"New value injection failed: {col} not found")

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_new_value",
                "cols": cols,
                "new_value": str(new_value),
                "prob": prob,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_new_value",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)
        return df_drift

    def inject_data_quality_issues(
        self,
        df: pd.DataFrame,
        issues: List[Dict],
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Orchestrates multiple data quality issues (drifts).
        """
        df_drift = df.copy()

        for issue in issues:
            method_name = issue.get("method")
            params = issue.get("params", {})

            # Inject df and context if not present
            if "df" not in params:
                params["df"] = df_drift
            if "time_col" not in params:
                params["time_col"] = time_col
            if "block_column" not in params and block_column:
                params["block_column"] = block_column
            if "auto_report" not in params:
                params["auto_report"] = (
                    False  # Don't report individual steps if orchestrating?
                )
                # Or maybe set to False and report at the end?

            if hasattr(self, method_name):
                method = getattr(self, method_name)
                try:
                    df_drift = method(**params)
                except Exception as e:
                    print(f"Failed to apply {method_name}: {e}")
            else:
                print(f"Method {method_name} not found")

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_data_quality_issues",
                "issues": issues,
                "generator_name": f"{gen_name}_data_quality",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)
        return df_drift

    # -------------------------
    def inject_nulls(
        self,
        df: pd.DataFrame,
        cols: List[str],
        prob: float = 0.1,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects Nulls/NaNs completely at random (MCAR).
        """
        return self.inject_missing_values_drift(
            df=df,
            feature_cols=cols,
            missing_fraction=prob,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

    # -------------------------
    def inject_label_shift(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_distribution: dict,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects label shift by resampling the target column to match a target distribution.

        Args:
            df: Input DataFrame.
            target_col: Target column to modify.
            target_distribution: Dict mapping label values to target proportions.
                                 Example: {0: 0.3, 1: 0.7} for 30% class 0, 70% class 1.
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with shifted label distribution.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        # Normalize target distribution
        total = sum(target_distribution.values())
        if total <= 0:
            raise ValueError("Target distribution values must sum to > 0")
        normalized_dist = {k: v / total for k, v in target_distribution.items()}

        # Calculate target counts for each label
        target_counts = {k: int(n * v) for k, v in normalized_dist.items()}

        # Adjust for rounding errors - add remainder to largest class
        remainder = n - sum(target_counts.values())
        if remainder > 0:
            max_label = max(normalized_dist, key=normalized_dist.get)
            target_counts[max_label] += remainder

        # Resample: assign labels to rows based on target distribution
        new_labels = []
        for label, count in target_counts.items():
            new_labels.extend([label] * count)

        # Shuffle and assign
        self.rng.shuffle(new_labels)
        df_drift.loc[rows, target_col] = new_labels[:n]

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_label_shift",
                "target_col": target_col,
                "target_distribution": target_distribution,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_label_shift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    # -------------------------
    # Target distribution drift
    # -------------------------
    def inject_concept_drift(
        self,
        df: pd.DataFrame,
        concept_drift_type: str = "label_flip",
        concept_drift_magnitude: float = 0.2,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        target_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects concept drift locally on selected rows.
        """
        if not target_column:
            raise ValueError("target_column must be provided for concept drift.")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        df_drift = df.copy()

        # Logic for drift types
        if concept_drift_type == "label_flip":
            # Flip a fraction of labels
            rows_to_flip = self.rng.choice(
                rows, size=int(len(rows) * concept_drift_magnitude), replace=False
            )

            # Assuming binary or categorical. If binary 0/1, just 1-x.
            # If categorical, pick another random category.
            unique_labels = df[target_column].unique()
            if len(unique_labels) == 2 and {0, 1}.issubset(
                unique_labels
            ):  # Binary numeric
                df_drift.loc[rows_to_flip, target_column] = (
                    1 - df_drift.loc[rows_to_flip, target_column]
                )
            else:
                # General categorical flip
                for r in rows_to_flip:
                    current_val = df_drift.at[r, target_column]
                    possible_vals = [v for v in unique_labels if v != current_val]
                    if possible_vals:
                        df_drift.at[r, target_column] = self.rng.choice(possible_vals)

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir

            drift_config = {
                "drift_method": "inject_concept_drift",
                "concept_drift_type": concept_drift_type,
                "magnitude": concept_drift_magnitude,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_concept_drift",
            }

            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)

        return df_drift

    # -------------------------
    # Orchestration / Legacy Support
    # -------------------------
    def inject_multiple_types_of_drift(
        self,
        df: pd.DataFrame,
        schedule: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        time_col: Optional[str] = None,
        block_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Applies a sequence of drift injections defined in a schedule.

        Args:
            df (pd.DataFrame): The dataframe to apply drift to.
            schedule (List[Dict[str, Any]]): A list of drift configurations.
                Each config must have a 'method' key (e.g., 'inject_feature_drift')
                and a 'params' dict.
            output_dir (Optional[str]): Override output directory.
            generator_name (Optional[str]): Override generator name.

            # Global Context Overrides (optional)
            time_col (Optional[str]): Timestamp column name.
            block_column (Optional[str]): Block column name.
            target_column (Optional[str]): Target column name.

        Returns:
            pd.DataFrame: The drifted dataframe.
        """
        current_df = df.copy()

        for i, config in enumerate(schedule):
            method_name = config.get("method")
            params = config.get("params", {}).copy()

            if not method_name or not hasattr(self, method_name):
                warnings.warn(
                    f"Unknown drift method '{method_name}' in schedule at index {i}. Skipping."
                )
                continue

            # Inject global overrides if not present in params
            if output_dir and "output_dir" not in params:
                params["output_dir"] = output_dir
            if generator_name and "generator_name" not in params:
                params["generator_name"] = f"{generator_name}_step_{i}"

            # Inject column context if not present
            if time_col and "time_col" not in params:
                params["time_col"] = time_col
            if block_column and "block_column" not in params:
                params["block_column"] = block_column
            if target_column and "target_column" not in params:
                # Some methods call use 'target_col' instead of 'target_column', handle both
                if "target_col" not in params:
                    params["target_col"] = target_column
                if "target_column" not in params:
                    params["target_column"] = target_column

            # Pass 'df' as the first argument (or keyword arg)
            # Most methods signature: method(df, ...)
            try:
                drift_method = getattr(self, method_name)
                # We pass current_df as the first argument 'df'
                # If params contains 'df', we remove it to avoid double argument
                if "df" in params:
                    del params["df"]

                current_df = drift_method(current_df, **params)
            except Exception as e:
                warnings.warn(f"Failed to apply drift '{method_name}': {e}")

        return current_df

    def inject_feature_drift_abrupt(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_type: str = "gaussian_noise",
        drift_magnitude: float = 0.2,
        change_index: int = 0,
        direction: str = "up",  # direction is unused in simple shift but kept for API compat
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Alias/Wrapper for injecting abrupt drift (step change) using inject_feature_drift.
        This corresponds to drifting all rows starting from `change_index`.
        """
        # "Abrupt" drift typically means a permanent change from a point onwards.
        # We can simulate this by setting start_index=change_index in inject_feature_drift.
        return self.inject_feature_drift(
            df=df,
            feature_cols=feature_cols,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            start_index=change_index,
            time_col=time_col,
            output_dir=output_dir,
            generator_name=generator_name,
        )

    # -------------------------
    # Binary Probabilistic Drift
    # -------------------------
    def inject_binary_probabilistic_drift(
        self,
        df: pd.DataFrame,
        target_col: str,
        probability: float = 0.4,
        noise_range: Tuple[float, float] = (0.1, 0.7),
        threshold: float = 0.5,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects probabilistic drift into a binary/boolean variable.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
            time_col=time_col,
        )

        df_drift = df.copy()

        # Simple probabilistic flip based on 'probability'
        # Or more complex logic if 'center/width' provided for gradual drift
        # For now, let's stick to the simpler version or use the gradual parameters if provided.

        # If gradual parameters are provided, calculate probabilities
        if center is not None and width is not None:
            # Sort rows for gradual progression
            rows = sorted(rows)
            if not rows:
                return df_drift

            probs = self._calculate_drift_probabilities(
                rows=rows,
                center=center,
                width=width,
                profile=profile,
                speed_k=speed_k,
                direction=direction,
                index_min=rows[0],
                index_max=rows[-1],
            )
            # Adjust probabilities: base prob * gradual factor?
            # Or replace base prob? Let's say probs is the probability of flip.
            # But the user also provided 'probability'.
            # Let's assume 'probability' is the max probability of flip.
            probs = probs * probability
        else:
            probs = np.full(len(rows), probability)

        for i, idx in enumerate(rows):
            p = probs[i]
            if self.rng.random() < p:
                # Add noise if float, or flip if binary
                curr_val = df_drift.at[idx, target_col]
                # Assuming binary 0/1 for simplification as per original intent
                if pd.api.types.is_numeric_dtype(df[target_col]) and set(
                    df[target_col].unique()
                ).issubset({0, 1}):
                    df_drift.at[idx, target_col] = 1 - curr_val
                else:
                    # If float (probabilistic output), add noise?
                    # The method name suggests binary drift.
                    pass
                    warnings.warn(
                        f"Skipping binary drift on non-binary column '{target_col}'"
                    )

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_binary_probabilistic_drift",
                "target_col": target_col,
                "probability": probability,
                "start_index": start_index,
                "block_index": block_index,
                "time_start": time_start,
                "generator_name": f"{gen_name}_binary_probabilistic_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config, time_col=time_col)
        return df_drift

    # -------------------------
    # Virtual Drift (Missing Values)
    # -------------------------
    def inject_missing_values_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        missing_fraction: float = 0.1,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Injects missing values (NaN) into specified columns.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )
        if len(rows) == 0:
            return df_drift

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Simple random injection for now, can be upgraded to windowed later if needed
            mask = self.rng.random(len(rows)) < missing_fraction
            target_indices = rows[mask]

            df_drift.loc[target_indices, col] = np.nan

        return df_drift

    # -------------------------
    # Covariate Shift
    # -------------------------
    def inject_correlation_matrix_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_correlation_matrix: np.ndarray,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects covariate drift by transforming numeric features to match a target correlation matrix.

        Uses Cholesky decomposition to transform standardized features to achieve the
        desired correlation structure.

        Args:
            df: Input DataFrame.
            feature_cols: List of numeric columns to transform.
            target_correlation_matrix: Target correlation matrix (n_features x n_features).
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with modified correlation structure.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        if len(rows) == 0:
            return df_drift

        # Validate feature columns
        valid_cols = [
            c
            for c in feature_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(valid_cols) == 0:
            warnings.warn("No valid numeric columns found for correlation drift")
            return df_drift

        # Validate target matrix shape
        n_features = len(valid_cols)
        if target_correlation_matrix.shape != (n_features, n_features):
            raise ValueError(
                f"Target matrix shape {target_correlation_matrix.shape} doesn't match "
                f"number of features ({n_features})"
            )

        # Ensure target matrix is positive semi-definite
        target_corr = self._ensure_psd_matrix(target_correlation_matrix)

        # Extract and standardize data
        X = df_drift.loc[rows, valid_cols].values.astype(float)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero
        X_std = (X - means) / stds

        # Cholesky decomposition of target correlation
        try:
            L_target = np.linalg.cholesky(target_corr)
        except np.linalg.LinAlgError:
            warnings.warn("Target matrix not positive definite, using adjusted version")
            target_corr = self._ensure_psd_matrix(
                target_corr + 0.01 * np.eye(n_features)
            )
            L_target = np.linalg.cholesky(target_corr)

        # Current correlation and its Cholesky
        current_corr = np.corrcoef(X_std, rowvar=False)
        if np.isnan(current_corr).any():
            current_corr = np.eye(n_features)
        current_corr = self._ensure_psd_matrix(current_corr)

        try:
            L_current = np.linalg.cholesky(current_corr)
            L_current_inv = np.linalg.inv(L_current)
        except np.linalg.LinAlgError:
            L_current_inv = np.eye(n_features)

        # Transform: X_new = X_std @ L_current_inv.T @ L_target.T
        X_transformed = X_std @ L_current_inv.T @ L_target.T

        # De-standardize
        X_final = X_transformed * stds + means

        df_drift.loc[rows, valid_cols] = X_final

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_correlation_matrix_drift",
                "feature_cols": valid_cols,
                "generator_name": f"{gen_name}_correlation_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    # -------------------------
    # New Category Drift
    # -------------------------
    def inject_new_category_drift(
        self,
        df: pd.DataFrame,
        feature_col: str,
        new_category: object,
        probability: float = 0.1,
        replace_categories: Optional[List] = None,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects a new category into a feature column.

        This drift simulates the emergence of a new category value that wasn't
        present in the original data (e.g., a new product type, region, etc.).

        Args:
            df: Input DataFrame.
            feature_col: Categorical column to modify.
            new_category: The new category value to introduce.
            probability: Probability of replacing existing values (0.0 to 1.0).
            replace_categories: Optional list of existing categories to replace.
                               If None, any existing category can be replaced.
            start_index, block_index, block_column: Selection parameters.
            time_start, time_end, time_ranges, specific_times: Time-based selection.
            center, width, profile, speed_k, direction: Gradual transition parameters.
            auto_report: Whether to generate a report.

        Returns:
            pd.DataFrame: DataFrame with new category injected.
        """
        if feature_col not in df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found")

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        # Calculate transition weights if gradual parameters provided
        if center is not None or width is not None:
            c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
            w_width = max(1, int(width if width is not None else max(1, n // 5)))
            w = self._window_weights(
                n,
                center=c,
                width=w_width,
                profile=profile,
                k=float(speed_k),
                direction=direction,
            )
            replace_probs = w * self._frac(probability)
        else:
            replace_probs = np.full(n, self._frac(probability))

        # Determine which rows to replace
        for i, idx in enumerate(rows):
            if self.rng.random() < replace_probs[i]:
                current_val = df_drift.at[idx, feature_col]

                # Only replace if current value is in replace_categories (if specified)
                if replace_categories is not None:
                    if current_val not in replace_categories:
                        continue

                df_drift.at[idx, feature_col] = new_category

        if self.auto_report:
            gen_name = generator_name or self.generator_name
            out_dir = output_dir or self.output_dir
            drift_config = {
                "drift_method": "inject_new_category_drift",
                "feature_col": feature_col,
                "new_category": str(new_category),
                "probability": probability,
                "generator_name": f"{gen_name}_new_category_drift",
            }
            if out_dir:
                df_drift.to_csv(
                    os.path.join(out_dir, f"{drift_config['generator_name']}.csv"),
                    index=False,
                )
                self._generate_reports(df, df_drift, drift_config)

        return df_drift

    # -------------------------
    # Binary Probabilistic Drift
    # -------------------------
    def inject_concept_drift_gradual(
        self,
        df: pd.DataFrame,
        concept_drift_type: str = "label_flip",
        concept_drift_magnitude: float = 0.2,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        blocks: Optional[Sequence] = None,
        block_start: Optional[object] = None,
        n_blocks: Optional[int] = None,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_ranges: Optional[Sequence[Tuple[str, str]]] = None,
        specific_times: Optional[Sequence[str]] = None,
        center: Optional[int] = None,
        width: Optional[int] = None,
        profile: str = "sigmoid",
        speed_k: float = 1.0,
        direction: str = "up",
        inconsistency: float = 0.0,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Injects probabilistic drift into a binary/boolean variable.

        Logic:
        1. Calculates a temporal weight 'w' (0 to 1) based on the window parameters (sigmoid, linear, etc.).
        2. For each eligible row, with probability p = w * probability:
           - Adds or subtracts a random noise value (from noise_range) to the current binary value (0 or 1).
           - e.g. NewValue = OldValue +/- Noise
        3. Re-binarizes the result: 1 if NewValue > threshold, else 0.

        Args:
            df: Input DataFrame.
            target_col: The binary column to modify.
            probability: The maximum probability that a modification occurs (when temporal weight w=1).
            noise_range: Tuple (min_noise, max_noise) to add/subtract.
            threshold: Threshold to decide the final 0 or 1.
            ... standard selection and window params ...
        """
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found.")

        df_drift = df.copy()

        # 1. Select Target Rows
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            blocks=blocks,
            block_start=block_start,
            n_blocks=n_blocks,
            time_start=time_start,
            time_end=time_end,
            time_ranges=time_ranges,
            specific_times=specific_times,
        )

        n = len(rows)
        if n == 0:
            return df_drift

        # 2. Compute Temporal Weights (w)
        c = int(n // 2) if center is None else int(np.clip(center, 0, n - 1))
        w_width = max(1, int(width if width is not None else max(1, n // 5)))

        w = self._window_weights(
            n,
            center=c,
            width=w_width,
            profile=profile,
            k=float(speed_k),
            direction=direction,
        )

        # 3. Apply Drift
        current_vals = df_drift.loc[rows, target_col].astype(float).values

        # Decide which rows are modified based on probability * w
        # random_draw < w * concept_drift_magnitude
        modification_mask = self.rng.random(n) < (w * concept_drift_magnitude)

        if np.any(modification_mask):
            # For label flipping/binary drift, we simulate it via noise + threshold
            # Default defaults for label flipping equivalent
            noise_range = (-1.0, 1.0)
            threshold = 0.5

            # Generate noise for all, but only use it where modification_mask is True
            noise = self.rng.uniform(noise_range[0], noise_range[1], size=n)

            # Decide sign: + or - (50% chance)
            signs = self.rng.choice([-1, 1], size=n)

            # Apply modifications
            deltas = signs * noise
            # Zero out deltas where we shouldn't modify
            deltas[~modification_mask] = 0.0

            new_vals_numeric = current_vals + deltas

            # Thresholding
            final_vals = (new_vals_numeric > threshold).astype(int)

            df_drift.loc[rows, target_col] = final_vals

    # -------------------------
    # Categorical & Boolean Drift
    # -------------------------
    def inject_categorical_frequency_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_magnitude: float = 0.3,
        perturbation: str = "uniform",  # 'uniform', 'invert', 'random'
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        generator_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Changes the frequency distribution of categories in a column.
        e.g. Makes rare categories more frequent, or inverts the distribution.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        for col in feature_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found")
                continue

            original_series = df.loc[rows, col]
            if original_series.empty:
                continue

            unique_vals = original_series.unique()
            if len(unique_vals) < 2:
                continue

            # Calculate target probabilities
            n_rows = len(rows)

            if perturbation == "uniform":
                # Tends towards uniform distribution (max entropy)
                probs = np.ones(len(unique_vals)) / len(unique_vals)
            elif perturbation == "random":
                probs = self.rng.random(len(unique_vals))
                probs /= probs.sum()
            elif perturbation == "invert":
                # Inverts current frequencies
                counts = original_series.value_counts(normalize=True).reindex(
                    unique_vals, fill_value=0
                )
                inv_counts = 1.0 - counts
                if inv_counts.sum() == 0:
                    probs = np.ones(len(unique_vals)) / len(unique_vals)
                else:
                    probs = inv_counts / inv_counts.sum()
            else:
                probs = np.ones(len(unique_vals)) / len(unique_vals)

            # Apply drift based on magnitude: mix original distribution with target distribution
            # Implementation: For 'magnitude' % of rows, resample from NEW distribution.

            mask = self.rng.random(n_rows) < drift_magnitude
            n_drift = mask.sum()

            if n_drift > 0:
                new_values = self.rng.choice(unique_vals, size=n_drift, p=probs)
                # Map back to dataframe indices
                drift_indices = rows[mask]
                df_drift.loc[drift_indices, col] = new_values

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_categorical_frequency_drift",
                "feature_cols": feature_cols,
                "drift_magnitude": drift_magnitude,
                "perturbation": perturbation,
                "generator_name": f"{self.generator_name}_freq_drift",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    def inject_typos_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_magnitude: float = 0.1,  # Probability of a row having a typo
        typo_density: int = 1,  # Number of typos per string
        typo_type: str = "random",  # 'swap', 'delete', 'duplicate', 'random'
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Injects typos into string columns.
        """

        def apply_typo(text, n_errors, method):
            if not isinstance(text, str) or len(text) < 2:
                return text

            chars = list(text)
            for _ in range(n_errors):
                if len(chars) < 2:
                    break
                idx = self.rng.integers(0, len(chars))

                op = method
                if op == "random":
                    op = self.rng.choice(["swap", "delete", "duplicate"])

                if op == "swap" and len(chars) > 1:
                    idx2 = (idx + 1) % len(chars)
                    chars[idx], chars[idx2] = chars[idx2], chars[idx]
                elif op == "delete":
                    chars.pop(idx)
                elif op == "duplicate":
                    chars.insert(idx, chars[idx])

            return "".join(chars)

        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Filter for rows to affect
            mask = self.rng.random(len(rows)) < drift_magnitude
            target_indices = rows[mask]

            if not target_indices.empty:
                # Vectorized apply is hard for random typos, using list comprehension
                original_vals = df_drift.loc[target_indices, col].astype(str).tolist()
                drifted_vals = [
                    apply_typo(x, typo_density, typo_type) for x in original_vals
                ]
                df_drift.loc[target_indices, col] = drifted_vals

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_typos_drift",
                "feature_cols": feature_cols,
                "drift_magnitude": drift_magnitude,
                "typo_type": typo_type,
                "generator_name": f"{self.generator_name}_typos",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    def inject_category_merge_drift(
        self,
        df: pd.DataFrame,
        col: str,
        categories_to_merge: List[Any],
        new_category_name: Any,
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Merges specific categories into one (e.g. 'Cat' + 'Dog' -> 'Pet').
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        if col in df.columns:
            mask = df_drift.loc[rows, col].isin(categories_to_merge)
            drift_indices = rows[mask]
            df_drift.loc[drift_indices, col] = new_category_name

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_category_merge_drift",
                "col": col,
                "merged": str(categories_to_merge),
                "new_name": str(new_category_name),
                "generator_name": f"{self.generator_name}_merge",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift

    def inject_boolean_drift(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        drift_magnitude: float = 0.3,  # Probability of flipping
        start_index: Optional[int] = None,
        block_index: Optional[int] = None,
        block_column: Optional[str] = None,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Specific drift for boolean columns: flips True <-> False.
        """
        df_drift = df.copy()
        rows = self._get_target_rows(
            df,
            start_index=start_index,
            block_index=block_index,
            block_column=block_column,
            time_col=time_col,
            **kwargs,
        )

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Check if boolean-like
            if not pd.api.types.is_bool_dtype(df[col]) and not set(
                df[col].dropna().unique()
            ).issubset({0, 1, 0.0, 1.0, True, False}):
                warnings.warn(
                    f"Column '{col}' does not appear to be boolean. Skipping."
                )
                continue

            mask = self.rng.random(len(rows)) < drift_magnitude
            flip_indices = rows[mask]

            if not flip_indices.empty:
                # Logical NOT for booleans
                if pd.api.types.is_bool_dtype(df[col]):
                    df_drift.loc[flip_indices, col] = ~df_drift.loc[flip_indices, col]
                else:
                    # For 0/1 integers
                    df_drift.loc[flip_indices, col] = (
                        1 - df_drift.loc[flip_indices, col]
                    )

        if self.auto_report:
            drift_config = {
                "drift_method": "inject_boolean_drift",
                "feature_cols": feature_cols,
                "drift_magnitude": drift_magnitude,
                "generator_name": f"{self.generator_name}_boolean",
            }
            self._generate_reports(df, df_drift, drift_config, time_col=self.time_col)

        return df_drift
