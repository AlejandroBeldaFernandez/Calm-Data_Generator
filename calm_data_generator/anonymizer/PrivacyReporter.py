"""
Privacy Reporter Module
Generates interactive reports comparing original data with privacy-preserved data.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger("PrivacyReporter")


class PrivacyReporter:
    """
    Generates privacy analysis reports comparing original and anonymized/private datasets.
    """

    @staticmethod
    def generate_privacy_report(
        original_df: pd.DataFrame,
        private_df: pd.DataFrame,
        output_dir: str,
        filename: str = "privacy_report.html",
        columns: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generates a comprehensive privacy report.

        Includes:
        - Distribution comparison (Overlay)
        - Correlation matrix difference
        - Utility loss metrics (MAE, etc.)
        - Uniqueness/Re-identification risk proxy
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping privacy report.")
            return None

        os.makedirs(output_dir, exist_ok=True)

        try:
            # 1. Identify common numeric columns
            if columns:
                num_cols = [
                    c
                    for c in columns
                    if c in original_df.columns
                    and c in private_df.columns
                    and pd.api.types.is_numeric_dtype(original_df[c])
                ]
            else:
                num_cols = [
                    c
                    for c in original_df.select_dtypes(include=[np.number]).columns
                    if c in private_df.columns
                ]

            # Limit columns for readability
            num_cols = num_cols[:10]

            # 2. Calculate Uniqueness (Risk Proxy)
            orig_unique = len(original_df.drop_duplicates()) / len(original_df) * 100
            priv_unique = len(private_df.drop_duplicates()) / len(private_df) * 100

            # 3. Calculate Correlation Diff (Utility Proxy)
            if len(num_cols) > 1:
                corr_orig = original_df[num_cols].corr()
                corr_priv = private_df[num_cols].corr()
                corr_diff = (corr_orig - corr_priv).abs()
                max_corr_diff = corr_diff.max().max()
            else:
                max_corr_diff = 0

            # 4. Create Subplots
            n_cols = min(3, len(num_cols))
            n_rows_dist = (len(num_cols) + n_cols - 1) // n_cols if num_cols else 0

            # Layout:
            # Row 1: Key Metrics (Indicators)
            # Row 2: Correlation Difference Heatmap (if applicable)
            # Rows 3+: Distribution plots

            specs = [
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ]
            row_titles = ["Privacy & Utility Metrics"]

            current_row = 2
            if len(num_cols) > 1:
                specs.append([{"colspan": 3, "type": "heatmap"}, None, None])
                row_titles.append("Correlation Structure Loss (Abs Diff)")
                current_row += 1

            for _ in range(n_rows_dist):
                specs.append([{"type": "xy"}] * 3)  # Assuming max 3 cols
                row_titles.append("Feature Distributions")

            fig = make_subplots(
                rows=current_row + n_rows_dist - 1,
                cols=3,
                specs=specs,
                subplot_titles=row_titles + num_cols,
                vertical_spacing=0.05,
            )

            # --- Row 1: Metrics ---
            # Uniqueness Change
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=priv_unique,
                    delta={
                        "reference": orig_unique,
                        "relative": False,
                        "valueformat": ".1f",
                    },
                    title={"text": "Uniqueness %"},
                    number={"suffix": "%"},
                    domain={"row": 0, "column": 0},
                ),
                row=1,
                col=1,
            )

            # Max Corr Diff
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=max_corr_diff,
                    title={"text": "Max Corr. Breakage"},
                    domain={"row": 0, "column": 1},
                ),
                row=1,
                col=2,
            )

            # Rows count (Utility Context)
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=len(private_df),
                    title={"text": "Rows Count"},
                    domain={"row": 0, "column": 2},
                ),
                row=1,
                col=3,
            )

            # --- Row 2: Correlation Heatmap ---
            heatmap_row_idx = 2
            if len(num_cols) > 1:
                fig.add_trace(
                    go.Heatmap(
                        z=corr_diff.values,
                        x=corr_diff.columns,
                        y=corr_diff.index,
                        colorscale="Reds",
                        zmin=0,
                        zmax=1,
                        colorbar=dict(title="Abs Diff"),
                    ),
                    row=2,
                    col=1,
                )
                heatmap_row_idx += 1

            # --- Row 3+: Distributions ---
            for idx, col in enumerate(num_cols):
                row = heatmap_row_idx + (idx // 3)
                col_pos = (idx % 3) + 1

                # Original
                fig.add_trace(
                    go.Histogram(
                        x=original_df[col],
                        name="Original",
                        marker_color="blue",
                        opacity=0.5,
                        showlegend=(idx == 0),
                        legendgroup="orig",
                    ),
                    row=row,
                    col=col_pos,
                )

                # Private
                fig.add_trace(
                    go.Histogram(
                        x=private_df[col],
                        name="Private",
                        marker_color="red",
                        opacity=0.5,
                        showlegend=(idx == 0),
                        legendgroup="priv",
                    ),
                    row=row,
                    col=col_pos,
                )

            fig.update_layout(
                title_text="Privacy Anonymization Report",
                height=300 + 300 + (250 * n_rows_dist),
                barmode="overlay",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate privacy report: {e}", exc_info=True)
            return None
