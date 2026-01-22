"""
Visualizer Module
Generates interactive Plotly HTML plots for data visualization.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger("Visualizer")


class Visualizer:
    """
    Generates interactive Plotly HTML visualizations for data reports.
    """

    @staticmethod
    def generate_density_plots(
        df: pd.DataFrame,
        output_dir: str,
        columns: Optional[List[str]] = None,
        filename: str = "density_plots.html",
        color_col: Optional[str] = None,
        comparison_df: Optional[pd.DataFrame] = None,
        labels: tuple = ("Generated", "Original"),
    ) -> Optional[str]:
        """
        Generates interactive density/distribution plots for numeric columns.

        Args:
            df: Primary DataFrame to plot.
            comparison_df: Optional second DataFrame for overlay comparison.
            labels: Labels for (primary, comparison) datasets.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping density plots.")
            return None

        try:
            # Select numeric columns
            if columns:
                numeric_cols = [
                    c
                    for c in columns
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                ]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                logger.warning("No numeric columns found for density plots.")
                return None

            # Limit to 12 columns max
            numeric_cols = numeric_cols[:12]

            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=numeric_cols,
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
            )

            for idx, col in enumerate(numeric_cols):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1

                # Primary dataset
                fig.add_trace(
                    go.Histogram(
                        x=df[col].dropna(),
                        name=labels[0],
                        opacity=0.7 if comparison_df is not None else 1.0,
                        marker_color="#636EFA",
                        showlegend=(idx == 0),
                        legendgroup="primary",
                        histnorm="probability density"
                        if comparison_df is not None
                        else None,
                    ),
                    row=row,
                    col=col_pos,
                )

                # Comparison dataset (if provided)
                if comparison_df is not None and col in comparison_df.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=comparison_df[col].dropna(),
                            name=labels[1],
                            opacity=0.7,
                            marker_color="#EF553B",
                            showlegend=(idx == 0),
                            legendgroup="comparison",
                            histnorm="probability density",
                        ),
                        row=row,
                        col=col_pos,
                    )

            title = "Feature Distributions"
            if comparison_df is not None:
                title = f"Distribution Comparison: {labels[0]} vs {labels[1]}"

            fig.update_layout(
                title_text=title,
                height=300 * n_rows,
                showlegend=(comparison_df is not None),
                barmode="overlay",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate density plots: {e}")
            return None

    @staticmethod
    def generate_dimensionality_plot(
        df: pd.DataFrame,
        output_dir: str,
        filename: str = "dimensionality_plot.html",
        color_col: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates combined PCA + UMAP visualization in a single HTML file.
        """
        if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Required libraries not available.")
            return None

        try:
            # Select numeric columns and drop NaN
            numeric_df = df.select_dtypes(include=[np.number]).dropna()

            if numeric_df.shape[1] < 2:
                logger.warning(
                    "Not enough numeric columns for dimensionality reduction."
                )
                return None

            if numeric_df.shape[0] < 10:
                logger.warning("Not enough samples for dimensionality reduction.")
                return None

            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            # === PCA ===
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)

            pca_df = pd.DataFrame(
                pca_result,
                columns=["PC1", "PC2"],
                index=numeric_df.index,
            )

            # === UMAP (if available and enough samples) ===
            umap_df = None
            if UMAP_AVAILABLE and numeric_df.shape[0] >= 15:
                try:
                    reducer = umap.UMAP(
                        n_neighbors=min(15, numeric_df.shape[0] - 1),
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                    )
                    umap_result = reducer.fit_transform(scaled_data)
                    umap_df = pd.DataFrame(
                        umap_result,
                        columns=["UMAP1", "UMAP2"],
                        index=numeric_df.index,
                    )
                except Exception as e:
                    logger.warning(f"UMAP failed: {e}")

            # Add color column if provided
            color_values = None
            if color_col and color_col in df.columns:
                color_values = df.loc[numeric_df.index, color_col].values

            # Create subplots
            if umap_df is not None:
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=[
                        f"PCA (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})",
                        "UMAP",
                    ],
                    horizontal_spacing=0.1,
                )

                # PCA scatter
                if color_values is not None:
                    for val in np.unique(color_values):
                        mask = color_values == val
                        fig.add_trace(
                            go.Scatter(
                                x=pca_df.loc[mask, "PC1"],
                                y=pca_df.loc[mask, "PC2"],
                                mode="markers",
                                name=str(val),
                                legendgroup=str(val),
                                showlegend=True,
                            ),
                            row=1,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=umap_df.loc[mask, "UMAP1"],
                                y=umap_df.loc[mask, "UMAP2"],
                                mode="markers",
                                name=str(val),
                                legendgroup=str(val),
                                showlegend=False,
                            ),
                            row=1,
                            col=2,
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=pca_df["PC1"],
                            y=pca_df["PC2"],
                            mode="markers",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=umap_df["UMAP1"],
                            y=umap_df["UMAP2"],
                            mode="markers",
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

                fig.update_xaxes(title_text="PC1", row=1, col=1)
                fig.update_yaxes(title_text="PC2", row=1, col=1)
                fig.update_xaxes(title_text="UMAP1", row=1, col=2)
                fig.update_yaxes(title_text="UMAP2", row=1, col=2)

            else:
                # Only PCA
                fig = go.Figure()
                if color_values is not None:
                    for val in np.unique(color_values):
                        mask = color_values == val
                        fig.add_trace(
                            go.Scatter(
                                x=pca_df.loc[mask, "PC1"],
                                y=pca_df.loc[mask, "PC2"],
                                mode="markers",
                                name=str(val),
                            )
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=pca_df["PC1"],
                            y=pca_df["PC2"],
                            mode="markers",
                            showlegend=False,
                        )
                    )

                fig.update_layout(
                    title=f"PCA Visualization (Explained Variance: {sum(pca.explained_variance_ratio_):.1%})",
                )

            fig.update_layout(height=500)

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate dimensionality plot: {e}")
            return None

    @staticmethod
    def generate_sdv_scores_card(
        overall_score: float,
        weighted_score: float,
        output_dir: str,
        filename: str = "sdv_scores.html",
    ) -> Optional[str]:
        """
        Generates a clean, simple HTML card showing SDV scores.
        """
        try:
            # Determine color based on score
            def get_color(score):
                if score >= 0.75:
                    return "#28a745"  # Green
                elif score >= 0.50:
                    return "#ffc107"  # Yellow
                else:
                    return "#dc3545"  # Red

            html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        .score-card {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            display: flex;
            gap: 40px;
            padding: 20px;
            justify-content: center;
        }}
        .score-box {{
            text-align: center;
            padding: 20px 40px;
            border-radius: 12px;
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-label {{
            font-size: 14px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <div class="score-card">
        <div class="score-box">
            <div class="score-label">Overall Quality</div>
            <div class="score-value" style="color: {get_color(overall_score)}">
                {overall_score:.1%}
            </div>
        </div>
        <div class="score-box">
            <div class="score-label">Weighted Quality</div>
            <div class="score-value" style="color: {get_color(weighted_score)}">
                {weighted_score:.1%}
            </div>
        </div>
    </div>
</body>
</html>
"""
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                f.write(html)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate SDV scores card: {e}")
            return None

    @staticmethod
    def generate_sdv_evolution_plot(
        scores: List[Dict[str, Any]],
        output_dir: str,
        filename: str = "sdv_evolution.html",
        x_labels: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generates SDV quality evolution plot showing overall and weighted scores.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping SDV evolution plot.")
            return None

        try:
            if not scores:
                logger.warning("No scores provided for SDV evolution plot.")
                return None

            overall_scores = [s.get("overall", 0) for s in scores]
            weighted_scores = [s.get("weighted", 0) for s in scores]

            if x_labels is None:
                x_labels = [f"Block {i + 1}" for i in range(len(scores))]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=overall_scores,
                    mode="lines+markers",
                    name="Overall SDV Score",
                    line=dict(color="blue", width=2),
                    marker=dict(size=10),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=weighted_scores,
                    mode="lines+markers",
                    name="Weighted SDV Score",
                    line=dict(color="green", width=2),
                    marker=dict(size=10),
                )
            )

            fig.update_layout(
                title="SDV Quality Evolution",
                xaxis_title="Block / Time Period",
                yaxis_title="Quality Score",
                yaxis=dict(range=[0, 1]),
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode="x unified",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate SDV evolution plot: {e}")
            return None

    @staticmethod
    def generate_drift_analysis(
        original_df: pd.DataFrame,
        drifted_df: pd.DataFrame,
        output_dir: str,
        filename: str = "drift_analysis.html",
        columns: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generates comprehensive drift analysis visualizations.

        Includes:
        - Drift configuration summary card
        - KS test p-values and Cohen's d effect size
        - JS Divergence bar chart
        - Duplicates percentage
        - Overlay density plots per feature
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping drift analysis.")
            return None

        try:
            from scipy.spatial.distance import jensenshannon
            from scipy.stats import ks_2samp

            # Select numeric columns
            if columns:
                numeric_cols = [
                    c
                    for c in columns
                    if c in original_df.columns
                    and c in drifted_df.columns
                    and pd.api.types.is_numeric_dtype(original_df[c])
                ]
            else:
                numeric_cols = [
                    c
                    for c in original_df.select_dtypes(include=[np.number]).columns
                    if c in drifted_df.columns
                ]

            if not numeric_cols:
                logger.warning("No numeric columns found for drift analysis.")
                return None

            # Limit to 12 columns
            numeric_cols = numeric_cols[:12]

            # Calculate metrics for each column
            metrics = {}
            for col in numeric_cols:
                try:
                    orig_vals = original_df[col].dropna().values
                    drift_vals = drifted_df[col].dropna().values

                    if len(orig_vals) == 0 or len(drift_vals) == 0:
                        continue

                    # JS Divergence
                    min_val = min(orig_vals.min(), drift_vals.min())
                    max_val = max(orig_vals.max(), drift_vals.max())
                    bins = np.linspace(min_val, max_val, 50)
                    hist_orig, _ = np.histogram(orig_vals, bins=bins, density=True)
                    hist_drift, _ = np.histogram(drift_vals, bins=bins, density=True)
                    hist_orig = (hist_orig + 1e-10) / (hist_orig + 1e-10).sum()
                    hist_drift = (hist_drift + 1e-10) / (hist_drift + 1e-10).sum()
                    js_div = jensenshannon(hist_orig, hist_drift)

                    # KS test
                    ks_stat, ks_pval = ks_2samp(orig_vals, drift_vals)

                    # Cohen's d
                    pooled_std = np.sqrt(
                        (orig_vals.std() ** 2 + drift_vals.std() ** 2) / 2
                    )
                    cohens_d = (
                        (drift_vals.mean() - orig_vals.mean()) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    metrics[col] = {
                        "js_div": js_div,
                        "ks_stat": ks_stat,
                        "ks_pval": ks_pval,
                        "cohens_d": cohens_d,
                        "orig_mean": orig_vals.mean(),
                        "drift_mean": drift_vals.mean(),
                        "pct_change": (
                            (drift_vals.mean() - orig_vals.mean())
                            / abs(orig_vals.mean())
                            * 100
                        )
                        if orig_vals.mean() != 0
                        else 0,
                    }
                except Exception:
                    metrics[col] = {
                        "js_div": 0,
                        "ks_stat": 0,
                        "ks_pval": 1,
                        "cohens_d": 0,
                    }

            # Calculate duplicates percentage
            try:
                common_cols = list(set(original_df.columns) & set(drifted_df.columns))
                orig_unique = original_df[common_cols].drop_duplicates()
                merged = drifted_df[common_cols].merge(
                    orig_unique, how="left", indicator=True
                )
                cross_dup_count = (merged["_merge"] == "both").sum()
                cross_dup_pct = (
                    cross_dup_count / len(drifted_df) * 100
                    if len(drifted_df) > 0
                    else 0
                )
            except Exception:
                cross_dup_pct = 0

            # Sort by JS divergence
            sorted_cols = sorted(
                metrics.keys(), key=lambda x: metrics[x]["js_div"], reverse=True
            )

            # Build HTML report
            html_parts = []

            # 1. Drift Configuration Summary Card
            drift_cols_str = ", ".join(columns) if columns else "All numeric"
            drift_type = (
                drift_config.get("drift_type", "N/A") if drift_config else "N/A"
            )
            drift_mag = (
                drift_config.get("drift_magnitude", "N/A") if drift_config else "N/A"
            )

            html_parts.append(f"""
            <div style="font-family: 'Segoe UI', sans-serif; padding: 20px; max-width: 1200px; margin: auto;">
                <h1 style="color: #333;">🌊 Drift Analysis Report</h1>
                
                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">
                    <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                        <div style="font-size: 14px; opacity: 0.8;">AFFECTED COLUMNS</div>
                        <div style="font-size: 24px; font-weight: bold;">{drift_cols_str}</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                        <div style="font-size: 14px; opacity: 0.8;">DRIFT TYPE</div>
                        <div style="font-size: 24px; font-weight: bold;">{drift_type}</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                        <div style="font-size: 14px; opacity: 0.8;">MAGNITUDE</div>
                        <div style="font-size: 24px; font-weight: bold;">{drift_mag}</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #fa709a, #fee140); color: white; padding: 20px; border-radius: 12px; flex: 1; min-width: 200px;">
                        <div style="font-size: 14px; opacity: 0.8;">DUPLICATES WITH ORIGINAL</div>
                        <div style="font-size: 24px; font-weight: bold;">{cross_dup_pct:.1f}%</div>
                    </div>
                </div>

                <h2>📊 Statistical Analysis by Feature</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Feature</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">JS Div</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">KS Stat</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">KS p-value</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Cohen's d</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Mean Δ%</th>
                    </tr>
            """)

            for col in sorted_cols:
                m = metrics[col]
                pval_color = "#28a745" if m["ks_pval"] > 0.05 else "#dc3545"
                js_color = (
                    "#dc3545"
                    if m["js_div"] > 0.1
                    else "#ffc107"
                    if m["js_div"] > 0.05
                    else "#28a745"
                )
                html_parts.append(f"""
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #dee2e6; font-weight: bold;">{col}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6; color: {js_color};">{m["js_div"]:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{m["ks_stat"]:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6; color: {pval_color};">{m["ks_pval"]:.4f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{m["cohens_d"]:+.3f}</td>
                        <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{m["pct_change"]:+.1f}%</td>
                    </tr>
                """)

            html_parts.append("</table>")

            # Close HTML wrapper
            html_parts.append("</div>")

            # Save stats HTML
            stats_path = os.path.join(output_dir, "drift_stats.html")
            with open(stats_path, "w") as f:
                f.write("".join(html_parts))

            # Generate Plotly density plots
            n_density_cols = min(6, len(sorted_cols))
            n_cols = min(3, n_density_cols)
            n_rows = (n_density_cols + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=sorted_cols[:n_density_cols],
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )

            for idx, col in enumerate(sorted_cols[:n_density_cols]):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1

                fig.add_trace(
                    go.Histogram(
                        x=original_df[col].dropna(),
                        name="Original",
                        opacity=0.6,
                        marker_color="#636EFA",
                        showlegend=(idx == 0),
                        legendgroup="original",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

                fig.add_trace(
                    go.Histogram(
                        x=drifted_df[col].dropna(),
                        name="Drifted",
                        opacity=0.6,
                        marker_color="#EF553B",
                        showlegend=(idx == 0),
                        legendgroup="drifted",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_pos,
                )

            fig.update_layout(
                title_text="Distribution Comparison: Original vs Drifted",
                height=300 * n_rows,
                barmode="overlay",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate drift analysis: {e}")
            return None

    @staticmethod
    def generate_sequence_plot(
        df: pd.DataFrame,
        entity_col: str,
        time_col: str,
        output_dir: str,
        filename: str = "sequence_plot.html",
        feature_cols: Optional[List[str]] = None,
        n_entities: int = 5,
    ) -> Optional[str]:
        """
        Generates line plots for sequential data, showing trajectories of a few entities.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping sequence plot.")
            return None

        try:
            # Validate columns
            if entity_col not in df.columns or time_col not in df.columns:
                logger.warning(
                    f"Entity col '{entity_col}' or Time col '{time_col}' not found."
                )
                return None

            # Select features
            if feature_cols:
                plot_cols = [
                    c
                    for c in feature_cols
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                ]
            else:
                plot_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                plot_cols = [
                    c for c in plot_cols if c not in [entity_col]
                ]  # Exclude ID if numeric

            if not plot_cols:
                logger.warning("No numeric features to plot for sequences.")
                return None

            # Select top N entities
            entities = df[entity_col].unique()[:n_entities]
            subset = df[df[entity_col].isin(entities)].copy()

            if subset.empty:
                logger.warning("No data found for the selected entities.")
                return None

            # Sort by time
            subset = subset.sort_values(by=[entity_col, time_col])

            # Create subplots
            n_cols = 1
            n_rows = min(len(plot_cols), 5)  # Max 5 features stacked
            plot_cols = plot_cols[:n_rows]

            fig = make_subplots(
                rows=n_rows,
                cols=1,
                subplot_titles=plot_cols,
                shared_xaxes=True,
                vertical_spacing=0.05,
            )

            for i, col in enumerate(plot_cols):
                for entity in entities:
                    mask = subset[entity_col] == entity
                    row_data = subset[mask]

                    fig.add_trace(
                        go.Scatter(
                            x=row_data[time_col],
                            y=row_data[col],
                            mode="lines+markers",
                            name=str(entity),
                            legendgroup=str(entity),
                            showlegend=(i == 0),  # Only show legend once
                            line=dict(width=1.5),
                            marker=dict(size=4),
                        ),
                        row=i + 1,
                        col=1,
                    )

            fig.update_layout(
                title=f"Sequence Trajectories (Top {len(entities)} Entities)",
                height=300 * n_rows,
                hovermode="x unified",
            )

            output_path = os.path.join(output_dir, filename)
            pio.write_html(fig, output_path, include_plotlyjs=True, full_html=True)
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate sequence plot: {e}")
            return None
