import pandas as pd
import numpy as np
from calm_data_generator.generators.tabular import RealGenerator
import pytest


def test_reporting_generation():
    """Explicitly test report generation with new dependencies."""
    # Create simple data
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, 100),
            "B": np.random.choice(["X", "Y"], 100),
            "C": np.random.randint(0, 100, 100),
        }
    )

    gen = RealGenerator(auto_report=True)

    # Generate report explicitly via internal method or by triggering generation with auto_report=True
    # We will use the public API generate() which triggers report if auto_report=True
    print("Starting generation with reporting...")
    gen.generate(df, n_samples=10, method="cart", target_col="B")
    print("Generation and reporting finished successfully.")


if __name__ == "__main__":
    test_reporting_generation()
