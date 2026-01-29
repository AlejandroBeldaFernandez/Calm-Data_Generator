#!/usr/bin/env python
"""
Comprehensive Test for Drift/Scenario Reporting Enhancements.
Tests DriftInjector, RealGenerator, and ScenarioInjector report generation.
"""

import pandas as pd
import numpy as np
import os
import shutil
from sklearn.datasets import load_iris

print("=" * 70)
print("COMPREHENSIVE DRIFT/SCENARIO REPORTING TEST")
print("=" * 70)

# Load Iris dataset for testing
iris = load_iris()
real_df = pd.DataFrame(iris.data, columns=iris.feature_names)
real_df["target"] = iris.target

output_base = "comprehensive_test_reports"
if os.path.exists(output_base):
    shutil.rmtree(output_base)
os.makedirs(output_base)

results = {}

# =============================================================================
# TEST 1: RealGenerator with drift_injection_config
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: RealGenerator with drift_injection_config")
print("=" * 70)

try:
    from calm_data_generator.generators.tabular.RealGenerator import RealGenerator

    output_dir_1 = os.path.join(output_base, "1_real_generator_drift")

    gen = RealGenerator(auto_report=True)
    synthetic_df = gen.generate(
        data=real_df,
        n_samples=100,
        method="tvae",
        output_dir=output_dir_1,
        drift_injection_config=[
            {
                "method": "inject_drift",
                "params": {
                    "columns": "sepal length (cm)",
                    "drift_type": "shift",
                    "magnitude": 0.3,
                    "mode": "abrupt",
                },
            }
        ],
    )

    files = os.listdir(output_dir_1)
    has_drift_stats = "drift_stats.html" in files
    has_plot_comparison = "plot_comparison.html" in files
    has_report_json = "report_results.json" in files

    print(f"   Files generated: {len(files)}")
    print(f"   ✅ drift_stats.html: {'YES' if has_drift_stats else 'NO'}")
    print(f"   ✅ plot_comparison.html: {'YES' if has_plot_comparison else 'NO'}")
    print(f"   ✅ report_results.json: {'YES' if has_report_json else 'NO'}")

    results["RealGenerator"] = (
        has_drift_stats and has_plot_comparison and has_report_json
    )

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    results["RealGenerator"] = False

# =============================================================================
# TEST 2: DriftInjector standalone
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: DriftInjector (standalone)")
print("=" * 70)

try:
    from calm_data_generator.generators.drift.DriftInjector import DriftInjector
    from calm_data_generator.generators.tabular.QualityReporter import QualityReporter

    output_dir_2 = os.path.join(output_base, "2_drift_injector")
    os.makedirs(output_dir_2, exist_ok=True)

    # Generate synthetic data first (simple copy with noise)
    synthetic_base = real_df.copy()
    for col in synthetic_base.select_dtypes(include=[np.number]).columns:
        if col != "target":
            synthetic_base[col] += np.random.randn(len(synthetic_base)) * 0.1

    # Apply drift using inject_drift API
    injector = DriftInjector()
    drifted_df = injector.inject_drift(
        df=synthetic_base.copy(),
        columns="petal length (cm)",
        drift_type="shift",
        magnitude=0.5,
        mode="abrupt",
    )

    # Generate report manually with drift_config
    reporter = QualityReporter(verbose=False)
    reporter.generate_comprehensive_report(
        real_df=real_df,
        synthetic_df=drifted_df,
        generator_name="DriftInjector_Test",
        output_dir=output_dir_2,
        drift_config={
            "drift_type": "Abrupt Shift",
            "drift_magnitude": 0.5,
            "affected_columns": "petal length (cm)",
        },
    )

    files = os.listdir(output_dir_2)
    has_drift_stats = "drift_stats.html" in files
    has_plot_comparison = "plot_comparison.html" in files

    print(f"   Files generated: {len(files)}")
    print(f"   ✅ drift_stats.html: {'YES' if has_drift_stats else 'NO'}")
    print(f"   ✅ plot_comparison.html: {'YES' if has_plot_comparison else 'NO'}")

    results["DriftInjector"] = has_drift_stats and has_plot_comparison

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    results["DriftInjector"] = False

# =============================================================================
# TEST 3: ScenarioInjector evolve_features
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: ScenarioInjector (evolve_features)")
print("=" * 70)

try:
    from calm_data_generator.generators.dynamics.ScenarioInjector import (
        ScenarioInjector,
    )

    output_dir_3 = os.path.join(output_base, "3_scenario_injector")

    # Create time-indexed data
    scenario_df = real_df.copy()
    scenario_df["time_index"] = np.arange(len(scenario_df))

    injector = ScenarioInjector(seed=42)

    evolution_config = {
        "sepal length (cm)": {"type": "linear", "slope": 0.02},
        "petal width (cm)": {"type": "sinusoidal", "period": 50, "amplitude": 0.1},
    }

    evolved_df = injector.evolve_features(
        df=scenario_df,
        evolution_config=evolution_config,
        time_col="time_index",
        output_dir=output_dir_3,
        auto_report=True,
        generator_name="ScenarioEvolution",
    )

    files = os.listdir(output_dir_3)
    has_drift_stats = "drift_stats.html" in files
    has_evolution_plot = "evolution_plot.html" in files
    has_plot_comparison = "plot_comparison.html" in files

    print(f"   Files generated: {len(files)}")
    print(f"   ✅ drift_stats.html: {'YES' if has_drift_stats else 'NO'}")
    print(f"   ✅ evolution_plot.html: {'YES' if has_evolution_plot else 'NO'}")
    print(f"   ✅ plot_comparison.html: {'YES' if has_plot_comparison else 'NO'}")

    results["ScenarioInjector"] = (
        has_drift_stats and has_evolution_plot and has_plot_comparison
    )

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    results["ScenarioInjector"] = False

# =============================================================================
# TEST 4: QualityReporter WITHOUT drift_config (should NOT have drift_stats)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: QualityReporter WITHOUT drift_config (no drift_stats expected)")
print("=" * 70)

try:
    from calm_data_generator.generators.tabular.QualityReporter import QualityReporter

    output_dir_4 = os.path.join(output_base, "4_no_drift")
    os.makedirs(output_dir_4, exist_ok=True)

    # Generate synthetic data (simple copy with noise)
    synthetic_clean = real_df.copy()
    for col in synthetic_clean.select_dtypes(include=[np.number]).columns:
        if col != "target":
            synthetic_clean[col] += np.random.randn(len(synthetic_clean)) * 0.05

    reporter = QualityReporter(verbose=False)
    reporter.generate_comprehensive_report(
        real_df=real_df,
        synthetic_df=synthetic_clean,
        generator_name="CleanSynthetic",
        output_dir=output_dir_4,
        # NO drift_config passed
    )

    files = os.listdir(output_dir_4)
    has_drift_stats = "drift_stats.html" in files
    has_plot_comparison = "plot_comparison.html" in files

    print(f"   Files generated: {len(files)}")
    print(
        f"   ❌ drift_stats.html should NOT exist: {'NO (correct)' if not has_drift_stats else 'YES (WRONG!)'}"
    )
    print(f"   ✅ plot_comparison.html: {'YES' if has_plot_comparison else 'NO'}")

    results["NoDriftConfig"] = (not has_drift_stats) and has_plot_comparison

except Exception as e:
    print(f"   ❌ ERROR: {e}")
    results["NoDriftConfig"] = False

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

all_passed = True
for test_name, passed in results.items():
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"   {test_name}: {status}")
    if not passed:
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("🎉 ALL TESTS PASSED!")
else:
    print("⚠️ SOME TESTS FAILED - Check details above")
print("=" * 70)

print(f"\nReports saved to: {os.path.abspath(output_base)}/")
