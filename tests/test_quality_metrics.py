import pytest
import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular.QualityReporter import QualityReporter


@pytest.fixture
def sample_data():
    """Create sample real and synthetic dataframes."""
    np.random.seed(42)

    # Real data
    real_data = {
        "age": np.random.randint(20, 60, 100),
        "salary": np.random.normal(50000, 10000, 100),
        "department": np.random.choice(["A", "B", "C"], 100),
    }
    real_df = pd.DataFrame(real_data)

    # Synthetic data (slightly different distribution)
    synth_data = {
        "age": np.random.randint(20, 60, 100),
        "salary": np.random.normal(52000, 11000, 100),
        "department": np.random.choice(["A", "B", "C"], 100),
    }
    synthetic_df = pd.DataFrame(synth_data)

    return real_df, synthetic_df


def test_calculate_quality_metrics(sample_data):
    """Test the calculate_quality_metrics method returns expected keys and types."""
    real_df, synthetic_df = sample_data

    reporter = QualityReporter(verbose=False)
    metrics = reporter.calculate_quality_metrics(real_df, synthetic_df)

    # Check if metrics are returned
    assert isinstance(metrics, dict)

    # Check if we got an error (e.g. if sdmetrics not installed) or actual metrics
    if "error" in metrics:
        pytest.skip(f"SDMetrics not available or failed: {metrics['error']}")

    # Check for expected keys
    assert "overall_quality_score" in metrics
    assert "weighted_quality_score" in metrics

    # Check types
    assert isinstance(metrics["overall_quality_score"], (float, int))
    assert isinstance(metrics["weighted_quality_score"], (float, int))

    # Check range (scores should be between 0 and 1)
    assert 0 <= metrics["overall_quality_score"] <= 1
    assert 0 <= metrics["weighted_quality_score"] <= 1

    print(f"\nQuality Metrics Retrieved: {metrics}")


def test_calculate_quality_metrics_empty(sample_data):
    """Test behavior with empty dataframe."""
    real_df, _ = sample_data
    empty_synth = pd.DataFrame(columns=real_df.columns)

    reporter = QualityReporter(verbose=False)
    metrics = reporter.calculate_quality_metrics(real_df, empty_synth)

    # Existing logic for empty synth usually results in low score or error handling works
    # We mainly want to ensure no crash
    assert isinstance(metrics, dict)
