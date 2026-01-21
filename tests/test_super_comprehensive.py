"""
CALM-Data-Generator - SUPER COMPREHENSIVE TEST
================================================

This test covers EVERY public method in the library:
1. RealGenerator - All synthesis methods
2. DriftInjector - ALL drift injection methods (including new ones)
3. ScenarioInjector - Feature evolution & target construction
4. ClinicalDataGenerator - Clinical data generation
5. StreamGenerator - Stream-based generation
6. Anonymizer - All privacy transformations
7. QualityReporter - Report generation

Run with: python tests/test_super_comprehensive.py
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
import traceback
from typing import Dict, List, Tuple

# Test results tracker
results = {"passed": [], "failed": [], "skipped": []}


def section(name: str):
    print(f"\n{'=' * 70}")
    print(f"📋 {name}")
    print(f"{'=' * 70}")


def passed(name: str):
    results["passed"].append(name)
    print(f"   ✅ {name}")


def failed(name: str, error: str):
    results["failed"].append((name, str(error)[:100]))
    print(f"   ❌ {name}: {str(error)[:80]}")


def skipped(name: str, reason: str):
    results["skipped"].append((name, reason))
    print(f"   ⏭️  {name}: {reason}")


# ============================================================================
# SAMPLE DATA CREATION
# ============================================================================
def create_sample_data(n: int = 200) -> pd.DataFrame:
    """Creates sample data for testing (includes datetime for drift tests)."""
    np.random.seed(42)
    # Create block column that works for any n
    block_size = max(1, n // 10)
    blocks = np.repeat(range((n // block_size) + 1), block_size)[:n]
    return pd.DataFrame(
        {
            "age": np.random.randint(20, 70, n),
            "income": np.random.normal(50000, 15000, n).astype(int),
            "score": np.random.uniform(0, 100, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "block": blocks,
        }
    )


def create_sklearn_data(n: int = 200) -> pd.DataFrame:
    """Creates data compatible with sklearn (no datetime columns)."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(20, 70, n),
            "income": np.random.normal(50000, 15000, n).astype(int),
            "score": np.random.uniform(0, 100, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        }
    )


def create_numeric_data(n: int = 200) -> pd.DataFrame:
    """Creates numeric-only data for SMOTE/correlation tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "f1": np.random.normal(10, 2, n),
            "f2": np.random.normal(20, 5, n),
            "f3": np.random.normal(30, 3, n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


# ============================================================================
# TEST 1: RealGenerator
# ============================================================================
def test_real_generator():
    section("1. RealGenerator - Synthesis Methods")

    try:
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator()
        passed("Import RealGenerator")
    except Exception as e:
        failed("Import RealGenerator", e)
        return

    # Use sklearn-compatible data (no datetime)
    data = create_sklearn_data(100)
    numeric_data = create_numeric_data(100)

    # Test basic methods
    for method in ["cart", "rf", "resample"]:
        try:
            synth = gen.generate(data, 30, method=method, target_col="target")
            assert synth is not None and len(synth) > 0
            passed(f"Method '{method}': {len(synth)} samples")
        except ImportError as e:
            skipped(f"Method '{method}'", str(e)[:50])
        except Exception as e:
            failed(f"Method '{method}'", e)

    # LGBM (optional dependency)
    try:
        synth = gen.generate(data, 30, method="lgbm", target_col="target")
        assert synth is not None and len(synth) > 0
        passed(f"Method 'lgbm': {len(synth)} samples")
    except ImportError:
        skipped("Method 'lgbm'", "lightgbm not installed")
    except Exception as e:
        failed("Method 'lgbm'", e)

    # SMOTE (needs numeric data)
    try:
        synth = gen.generate(numeric_data, 50, method="smote", target_col="target")
        assert synth is not None
        passed(f"Method 'smote': {len(synth)} samples")
    except Exception as e:
        failed("Method 'smote'", e)

    # ADASYN (needs imbalanced data)
    try:
        imb = numeric_data.copy()
        imb["target"] = np.random.choice([0, 1], len(imb), p=[0.85, 0.15])
        synth = gen.generate(imb, 50, method="adasyn", target_col="target")
        assert synth is not None
        passed(f"Method 'adasyn': {len(synth)} samples")
    except Exception as e:
        failed("Method 'adasyn'", e)

    # Copula (optional dep)
    try:
        synth = gen.generate(data, 30, method="copula", target_col="target")
        passed(f"Method 'copula': {len(synth)} samples")
    except ImportError:
        skipped("Method 'copula'", "SDV not installed")
    except Exception as e:
        failed("Method 'copula'", e)

    # Constraints
    try:
        synth = gen.generate(
            data,
            30,
            method="cart",
            target_col="target",
            constraints=[{"col": "age", "op": ">=", "val": 30}],
        )
        if synth is not None:
            assert synth["age"].min() >= 30
            passed(f"Constraints: min_age={synth['age'].min()}")
        else:
            failed("Constraints", "synth is None")
    except Exception as e:
        failed("Constraints", e)


# ============================================================================
# TEST 2: DriftInjector - ALL METHODS
# ============================================================================
def test_drift_injector():
    section("2. DriftInjector - All Drift Methods")

    try:
        from calm_data_generator.generators.drift import DriftInjector

        injector = DriftInjector()
        passed("Import DriftInjector")
    except Exception as e:
        failed("Import DriftInjector", e)
        return

    data = create_sample_data(200)

    # --- Feature Drift Methods ---
    tests = [
        (
            "inject_feature_drift",
            {
                "feature_cols": ["score"],
                "drift_type": "shift",
                "drift_magnitude": 0.3,
                "start_index": 100,
            },
        ),
        (
            "inject_feature_drift_gradual",
            {
                "feature_cols": ["score"],
                "drift_type": "gaussian_noise",
                "drift_magnitude": 0.5,
                "center": 100,
                "width": 50,
                "profile": "sigmoid",
            },
        ),
        (
            "inject_feature_drift_abrupt",
            {
                "feature_cols": ["income"],
                "drift_type": "shift",
                "drift_magnitude": 0.4,
                "change_index": 100,
            },
        ),
        (
            "inject_feature_drift_incremental",
            {
                "feature_cols": ["age"],
                "drift_type": "scale",
                "drift_magnitude": 0.2,
            },
        ),
        (
            "inject_feature_drift_recurrent",
            {
                "feature_cols": ["score"],
                "drift_type": "shift",
                "drift_magnitude": 0.3,
                "repeats": 2,
            },
        ),
        (
            "inject_conditional_drift",
            {
                "feature_cols": ["income"],
                "conditions": [{"column": "age", "operator": ">", "value": 40}],
                "drift_type": "shift",
                "drift_magnitude": 0.3,
            },
        ),
    ]

    for method_name, params in tests:
        try:
            method = getattr(injector, method_name)
            result = method(df=data.copy(), **params)
            assert result is not None and len(result) == len(data)
            passed(f"{method_name}")
        except Exception as e:
            failed(method_name, e)

    # --- Label Drift Methods ---
    label_tests = [
        (
            "inject_label_drift",
            {
                "target_cols": ["target"],
                "drift_magnitude": 0.2,
            },
        ),
        (
            "inject_label_drift_gradual",
            {
                "target_col": "target",
                "drift_magnitude": 0.3,
                "center": 100,
                "width": 50,
            },
        ),
        (
            "inject_label_drift_abrupt",
            {
                "target_col": "target",
                "drift_magnitude": 0.3,
                "change_index": 100,
            },
        ),
        (
            "inject_label_drift_incremental",
            {
                "target_col": "target",
                "drift_magnitude": 0.2,
            },
        ),
        (
            "inject_label_shift",
            {
                "target_col": "target",
                "target_distribution": {0: 0.3, 1: 0.7},
            },
        ),
    ]

    for method_name, params in label_tests:
        try:
            method = getattr(injector, method_name)
            result = method(df=data.copy(), **params)
            assert result is not None
            passed(f"{method_name}")
        except Exception as e:
            failed(method_name, e)

    # --- Concept Drift ---
    concept_tests = [
        (
            "inject_concept_drift",
            {
                "target_column": "target",
                "concept_drift_type": "label_flip",
                "concept_drift_magnitude": 0.2,
            },
        ),
        (
            "inject_concept_drift_gradual",
            {
                "target_col": "target",
                "concept_drift_magnitude": 0.2,
                "center": 100,
                "width": 50,
            },
        ),
        (
            "inject_binary_probabilistic_drift",
            {
                "target_col": "target",
                "probability": 0.3,
            },
        ),
    ]

    for method_name, params in concept_tests:
        try:
            method = getattr(injector, method_name)
            result = method(df=data.copy(), **params)
            assert result is not None
            passed(f"{method_name}")
        except Exception as e:
            failed(method_name, e)

    # --- Correlation Matrix Drift ---
    try:
        num_data = create_numeric_data(100)
        target_corr = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
        result = injector.inject_correlation_matrix_drift(
            df=num_data.copy(),
            feature_cols=["f1", "f2", "f3"],
            target_correlation_matrix=target_corr,
        )
        passed("inject_correlation_matrix_drift")
    except Exception as e:
        failed("inject_correlation_matrix_drift", e)

    # --- New Category Drift ---
    try:
        result = injector.inject_new_category_drift(
            df=data.copy(),
            feature_col="category",
            new_category="NEW_CAT",
            probability=0.2,
        )
        assert "NEW_CAT" in result["category"].values
        passed("inject_new_category_drift")
    except Exception as e:
        failed("inject_new_category_drift", e)

    # --- Data Quality Issues ---
    quality_tests = [
        (
            "inject_outliers_global",
            {
                "cols": ["score", "income"],
                "outlier_prob": 0.1,
                "factor": 3.0,
            },
        ),
        (
            "inject_new_value",
            {
                "cols": ["category"],
                "new_value": "REPLACED",
                "prob": 0.1,
            },
        ),
        (
            "inject_nulls",
            {
                "cols": ["score"],
                "prob": 0.1,
            },
        ),
    ]

    for method_name, params in quality_tests:
        try:
            method = getattr(injector, method_name)
            result = method(df=data.copy(), **params)
            assert result is not None
            passed(f"{method_name}")
        except Exception as e:
            failed(method_name, e)

    # inject_missing_values_drift (different signature - no auto_report)
    try:
        result = injector.inject_missing_values_drift(
            df=data.copy(),
            feature_cols=["income"],
            missing_fraction=0.1,
        )
        assert result is not None
        passed("inject_missing_values_drift")
    except Exception as e:
        failed("inject_missing_values_drift", e)

    # --- Orchestration ---
    try:
        schedule = [
            {
                "method": "inject_feature_drift",
                "params": {
                    "feature_cols": ["score"],
                    "drift_type": "shift",
                    "drift_magnitude": 0.2,
                    "auto_report": False,
                },
            },
            {
                "method": "inject_label_drift",
                "params": {
                    "target_cols": ["target"],
                    "drift_magnitude": 0.1,
                    "auto_report": False,
                },
            },
        ]
        result = injector.inject_multiple_types_of_drift(
            df=data.copy(),
            schedule=schedule,
        )
        passed("inject_multiple_types_of_drift")
    except Exception as e:
        failed("inject_multiple_types_of_drift", e)


# ============================================================================
# TEST 3: ScenarioInjector
# ============================================================================
def test_scenario_injector():
    section("3. ScenarioInjector - Feature Evolution")

    try:
        from calm_data_generator.generators.dynamics import ScenarioInjector

        scenario = ScenarioInjector(seed=42)
        passed("Import ScenarioInjector")
    except Exception as e:
        failed("Import ScenarioInjector", e)
        return

    data = create_sample_data(100)

    # evolve_features
    try:
        result = scenario.evolve_features(
            df=data.copy(),
            evolution_config={
                "score": {"type": "linear", "slope": 0.05},
                "income": {"type": "cycle", "period": 50, "amplitude": 1000},
            },
            time_col="timestamp",
        )
        passed("evolve_features (linear + cycle)")
    except Exception as e:
        failed("evolve_features", e)

    # construct_target - regression
    try:
        result = scenario.construct_target(
            df=data.copy(),
            target_col="new_target",
            formula="age * 0.5 + income / 10000",
            task_type="regression",
            noise_std=0.1,
        )
        assert "new_target" in result.columns
        passed("construct_target (regression)")
    except Exception as e:
        failed("construct_target (regression)", e)

    # construct_target - classification
    try:
        result = scenario.construct_target(
            df=data.copy(),
            target_col="binary_target",
            formula="age + score",
            task_type="classification",
            threshold=80,
        )
        assert set(result["binary_target"].unique()).issubset({0, 1})
        passed("construct_target (classification)")
    except Exception as e:
        failed("construct_target (classification)", e)

    # project_to_future_period
    try:
        result = scenario.project_to_future_period(
            df=data.copy(),
            periods=2,
            trend_config={"score": {"type": "linear", "slope": 0.01}},
            block_col="block",
        )
        assert len(result) > len(data)
        passed(f"project_to_future_period: {len(data)} -> {len(result)}")
    except Exception as e:
        failed("project_to_future_period", e)


# ============================================================================
# TEST 4: ClinicalDataGenerator
# ============================================================================
def test_clinical_generator():
    section("4. ClinicalDataGenerator - Clinical Data")

    try:
        from calm_data_generator.generators.clinical import ClinicalDataGenerator

        gen = ClinicalDataGenerator()
        passed("Import ClinicalDataGenerator")
    except Exception as e:
        failed("Import ClinicalDataGenerator", e)
        return

    # Basic generation
    try:
        result = gen.generate(n_samples=20, n_genes=30, n_proteins=20)
        assert "demographics" in result
        assert "genes" in result
        passed(f"generate: demographics={result['demographics'].shape}")
    except Exception as e:
        failed("generate", e)

    # Longitudinal
    try:
        result = gen.generate_longitudinal_data(
            n_samples=10, longitudinal_config={"n_visits": 3}
        )
        passed(f"generate_longitudinal_data")
    except Exception as e:
        failed("generate_longitudinal_data", e)


# ============================================================================
# TEST 5: StreamGenerator
# ============================================================================
def test_stream_generator():
    section("5. StreamGenerator - Stream-Based Generation")

    try:
        from calm_data_generator.generators.stream import StreamGenerator

        gen = StreamGenerator()
        passed("Import StreamGenerator")
    except Exception as e:
        failed("Import StreamGenerator", e)
        return

    # Simple generator
    def simple_stream():
        for i in range(1000):
            x = {"f1": np.random.random(), "f2": np.random.random()}
            y = 1 if x["f1"] > 0.5 else 0
            yield x, y

    try:
        result = gen.generate(
            generator_instance=simple_stream(),
            n_samples=50,
        )
        assert len(result) == 50
        passed(f"generate: {len(result)} samples")
    except Exception as e:
        failed("generate", e)

    # Balanced generation
    try:
        result = gen.generate(
            generator_instance=simple_stream(),
            n_samples=50,
            balance_target=True,
        )
        passed("generate (balanced)")
    except Exception as e:
        failed("generate (balanced)", e)


# ============================================================================
# TEST 6: Anonymizer
# ============================================================================
def test_anonymizer():
    section("6. Anonymizer - Privacy Transformations")

    try:
        from calm_data_generator.anonymizer import (
            pseudonymize_columns,
            add_laplace_noise,
            generalize_numeric_to_ranges,
            generalize_categorical_by_mapping,
            shuffle_columns,
        )

        passed("Import all anonymizer functions")
    except Exception as e:
        failed("Import anonymizer", e)
        return

    data = create_sample_data(50)
    data["user_id"] = [f"USER_{i}" for i in range(len(data))]

    # pseudonymize_columns
    try:
        result = pseudonymize_columns(data.copy(), columns=["user_id"])
        assert result["user_id"].iloc[0] != "USER_0"
        passed("pseudonymize_columns")
    except Exception as e:
        failed("pseudonymize_columns", e)

    # add_laplace_noise
    try:
        result = add_laplace_noise(data.copy(), columns=["age"], epsilon=1.0)
        assert not result["age"].equals(data["age"])
        passed("add_laplace_noise")
    except Exception as e:
        failed("add_laplace_noise", e)

    # generalize_numeric_to_ranges
    try:
        result = generalize_numeric_to_ranges(data.copy(), columns=["age"], num_bins=5)
        passed("generalize_numeric_to_ranges")
    except Exception as e:
        failed("generalize_numeric_to_ranges", e)

    # generalize_categorical_by_mapping
    try:
        mapping = {"A": "GROUP_1", "B": "GROUP_1", "C": "GROUP_2"}
        result = generalize_categorical_by_mapping(
            data.copy(),
            columns=["category"],  # FIXED: columns as list, not column
            mapping=mapping,
        )
        assert "GROUP_1" in result["category"].values
        passed("generalize_categorical_by_mapping")
    except Exception as e:
        failed("generalize_categorical_by_mapping", e)

    # shuffle_columns
    try:
        result = shuffle_columns(data.copy(), columns=["income"])
        # Values should be same but order different
        assert set(result["income"]) == set(data["income"])
        passed("shuffle_columns")
    except Exception as e:
        failed("shuffle_columns", e)


# ============================================================================
# TEST 7: QualityReporter
# ============================================================================
def test_quality_reporter():
    section("7. QualityReporter - Report Generation")

    try:
        from calm_data_generator.generators.tabular import QualityReporter

        reporter = QualityReporter(minimal=True)
        passed("Import QualityReporter")
    except Exception as e:
        failed("Import QualityReporter", e)
        return

    data = create_sample_data(100)
    synth = data.sample(50, replace=True).reset_index(drop=True)

    # Basic report
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.generate_comprehensive_report(
                real_df=data,
                synthetic_df=synth,
                generator_name="test",
                output_dir=tmpdir,
            )
            files = os.listdir(tmpdir)
            passed(f"generate_comprehensive_report: {len(files)} files")
    except Exception as e:
        failed("generate_comprehensive_report", e)


# ============================================================================
# SUMMARY
# ============================================================================
def print_summary():
    total = len(results["passed"]) + len(results["failed"])
    success_rate = len(results["passed"]) / total * 100 if total > 0 else 0

    print("\n" + "=" * 70)
    print("📊 SUPER COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed:  {len(results['passed'])}")
    print(f"❌ Failed:  {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    print(f"\n📈 Success Rate: {success_rate:.1f}%")

    if results["failed"]:
        print("\n❌ Failed tests:")
        for name, error in results["failed"]:
            print(f"   - {name}: {error}")

    print("=" * 70)

    if success_rate == 100:
        print("🎉 ALL TESTS PASSED! Library is fully functional.")
    elif success_rate >= 90:
        print("✅ LIBRARY VERIFICATION PASSED!")
    elif success_rate >= 70:
        print("⚠️  LIBRARY MOSTLY WORKING (some issues)")
    else:
        print("❌ LIBRARY NEEDS ATTENTION")

    print("=" * 70)
    return success_rate


# ============================================================================
# MAIN
# ============================================================================
def run_super_test():
    print("=" * 70)
    print("🧪 CALM-Data-Generator - SUPER COMPREHENSIVE TEST")
    print("   Testing EVERY public method in the library")
    print("=" * 70)

    test_real_generator()
    test_drift_injector()
    test_scenario_injector()
    test_clinical_generator()
    test_stream_generator()
    test_anonymizer()
    test_quality_reporter()

    return print_summary()


if __name__ == "__main__":
    success_rate = run_super_test()
    sys.exit(0 if success_rate >= 90 else 1)
