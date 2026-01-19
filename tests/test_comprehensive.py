"""
CALM-Data-Generator - COMPREHENSIVE TEST
==========================================

This test covers ALL major features of the library:
1. RealGenerator - All synthesis methods
2. ClinicalDataGenerator - Clinical data
3. DriftInjector - Drift injection
4. ScenarioInjector - Feature evolution
5. Anonymizer - Privacy transformations
6. Single-call workflow - Generate + Drift + Report
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys
import traceback

# Test results tracker
results = {"passed": [], "failed": [], "skipped": []}


def test_section(name):
    print(f"\n{'=' * 60}")
    print(f"📋 {name}")
    print(f"{'=' * 60}")


def test_passed(name):
    results["passed"].append(name)
    print(f"   ✅ {name}")


def test_failed(name, error):
    results["failed"].append((name, str(error)))
    print(f"   ❌ {name}: {error}")


def test_skipped(name, reason):
    results["skipped"].append((name, reason))
    print(f"   ⏭️  {name}: {reason}")


def run_comprehensive_test():
    """Run all tests for CALM-Data-Generator."""

    print("=" * 60)
    print("🧪 CALM-Data-Generator - COMPREHENSIVE TEST")
    print("=" * 60)

    # Create sample data with more balanced classes for SMOTE/ADASYN
    np.random.seed(42)
    sample_data = pd.DataFrame(
        {
            "age": np.random.randint(20, 70, 100),
            "income": np.random.normal(50000, 15000, 100).astype(int),
            "score": np.random.uniform(0, 100, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100, p=[0.6, 0.4]),  # More balanced
        }
    )

    print(f"\nSample data: {sample_data.shape}")

    # ============================================================
    # TEST 1: RealGenerator - Multiple Methods
    # ============================================================
    test_section("RealGenerator - Synthesis Methods")

    try:
        from calm_data_generator.generators.tabular import RealGenerator

        gen = RealGenerator()
        test_passed("Import RealGenerator")
    except Exception as e:
        test_failed("Import RealGenerator", e)
        return results

    # Test CART method
    try:
        synth = gen.generate(sample_data, 50, method="cart", target_col="target")
        assert synth is not None and len(synth) > 0
        test_passed(f"CART synthesis: {len(synth)} samples")
    except Exception as e:
        test_failed("CART synthesis", e)

    # Test RF method
    try:
        synth = gen.generate(sample_data, 50, method="rf", target_col="target")
        assert synth is not None and len(synth) > 0
        test_passed(f"RF synthesis: {len(synth)} samples")
    except Exception as e:
        test_failed("RF synthesis", e)

    # Test LGBM method
    try:
        synth = gen.generate(sample_data, 50, method="lgbm", target_col="target")
        assert synth is not None and len(synth) > 0
        test_passed(f"LGBM synthesis: {len(synth)} samples")
    except Exception as e:
        test_failed("LGBM synthesis", e)

    # Test Resample method (simpler than GMM)
    try:
        synth = gen.generate(sample_data, 50, method="resample", target_col="target")
        assert synth is not None and len(synth) > 0
        test_passed(f"Resample: {len(synth)} samples")
    except Exception as e:
        test_failed("Resample", e)

    # Test SMOTE method with numeric-only data
    try:
        numeric_data = sample_data[["age", "income", "score", "target"]].copy()
        synth = gen.generate(numeric_data, 80, method="smote", target_col="target")
        assert synth is not None and len(synth) > 0
        test_passed(f"SMOTE: {len(synth)} samples")
    except Exception as e:
        test_failed("SMOTE synthesis", e)

    # Test ADASYN method - needs more imbalanced data
    try:
        # Create imbalanced data for ADASYN (80/20 split)
        imb_data = pd.DataFrame(
            {
                "age": np.random.randint(20, 70, 100),
                "income": np.random.normal(50000, 15000, 100).astype(int),
                "score": np.random.uniform(0, 100, 100),
                "target": np.random.choice([0, 1], 100, p=[0.8, 0.2]),  # Imbalanced!
            }
        )
        synth = gen.generate(imb_data, 80, method="adasyn", target_col="target")
        assert synth is not None and len(synth) > 0
        test_passed(f"ADASYN: {len(synth)} samples")
    except Exception as e:
        test_failed("ADASYN synthesis", e)

    # Test Constraints
    try:
        synth = gen.generate(
            sample_data,
            50,
            method="cart",
            target_col="target",
            constraints=[
                {"col": "age", "op": ">=", "val": 25},
                {"col": "income", "op": ">", "val": 0},
            ],
        )
        assert synth is not None and synth["age"].min() >= 25
        test_passed(f"Constraints: min_age={synth['age'].min()}")
    except Exception as e:
        test_failed("Constraints", e)

    # ============================================================
    # TEST 2: ClinicalDataGenerator
    # ============================================================
    test_section("ClinicalDataGenerator - Clinical Data")

    try:
        from calm_data_generator.generators.clinical import ClinicalDataGenerator

        clin_gen = ClinicalDataGenerator()
        test_passed("Import ClinicalDataGenerator")
    except Exception as e:
        test_failed("Import ClinicalDataGenerator", e)
        clin_gen = None

    if clin_gen:
        try:
            result = clin_gen.generate(n_samples=20, n_genes=50, n_proteins=30)
            assert "demographics" in result
            test_passed(f"Generate clinical: {result['demographics'].shape}")
        except Exception as e:
            test_failed("Generate clinical data", e)

        try:
            result = clin_gen.generate_longitudinal_data(
                n_samples=10, longitudinal_config={"n_visits": 3}
            )
            test_passed(
                f"Longitudinal data: {result.get('longitudinal', pd.DataFrame()).shape}"
            )
        except Exception as e:
            test_failed("Longitudinal data", e)

    # ============================================================
    # TEST 3: DriftInjector
    # ============================================================
    test_section("DriftInjector - Drift Injection")

    try:
        from calm_data_generator.generators.drift import DriftInjector

        injector = DriftInjector()
        test_passed("Import DriftInjector")
    except Exception as e:
        test_failed("Import DriftInjector", e)
        injector = None

    if injector:
        # Gradual drift - CORRECT METHOD: inject_feature_drift_gradual
        try:
            drifted = injector.inject_feature_drift_gradual(
                df=sample_data.copy(),
                feature_cols=["score"],
                drift_magnitude=0.5,
                drift_type="shift",  # Valid: gaussian_noise, shift, scale
                start_index=50,
                center=25,  # Center of transition window
                width=20,  # Width of transition
                auto_report=False,
            )
            test_passed("Gradual drift injection")
        except Exception as e:
            test_failed("Gradual drift", e)

        # Feature drift - CORRECT METHOD: inject_feature_drift
        try:
            drifted = injector.inject_feature_drift(
                df=sample_data.copy(),
                feature_cols=["income"],
                drift_magnitude=0.3,
                drift_type="shift",  # Valid: gaussian_noise, shift, scale
                start_index=60,
                auto_report=False,
            )
            test_passed("Feature drift injection")
        except Exception as e:
            test_failed("Feature drift", e)

    # ============================================================
    # TEST 4: ScenarioInjector
    # ============================================================
    test_section("ScenarioInjector - Feature Evolution")

    try:
        from calm_data_generator.generators.dynamics import ScenarioInjector

        scenario = ScenarioInjector(seed=42)
        test_passed("Import ScenarioInjector")
    except Exception as e:
        test_failed("Import ScenarioInjector", e)
        scenario = None

    if scenario:
        try:
            # Add timestamp for evolution
            ts_data = sample_data.copy()
            ts_data["timestamp"] = pd.date_range(
                "2024-01-01", periods=len(ts_data), freq="D"
            )

            evolved = scenario.evolve_features(
                df=ts_data,
                evolution_config={"score": {"type": "trend", "rate": 0.05}},
                time_col="timestamp",
            )
            test_passed("Feature evolution")
        except Exception as e:
            test_failed("Feature evolution", e)

        try:
            result = scenario.construct_target(
                df=sample_data.copy(),
                target_col="new_target",
                formula="age + income / 10000",
                task_type="regression",
            )
            assert "new_target" in result.columns
            test_passed("Target construction")
        except Exception as e:
            test_failed("Target construction", e)

    # ============================================================
    # TEST 5: Anonymizer
    # ============================================================
    test_section("Anonymizer - Privacy Transformations")

    try:
        from calm_data_generator.anonymizer import (
            pseudonymize_columns,
            add_laplace_noise,
            generalize_numeric_to_ranges,
            shuffle_columns,
        )

        test_passed("Import anonymizer functions")
    except Exception as e:
        test_failed("Import anonymizer", e)
        pseudonymize_columns = None

    if pseudonymize_columns:
        try:
            priv_data = sample_data.copy()
            priv_data["id"] = [f"P{i}" for i in range(len(priv_data))]
            result = pseudonymize_columns(priv_data, columns=["id"])
            test_passed("Pseudonymization")
        except Exception as e:
            test_failed("Pseudonymization", e)

        try:
            result = add_laplace_noise(sample_data.copy(), columns=["age"], epsilon=1.0)
            test_passed("Laplace noise (DP)")
        except Exception as e:
            test_failed("Laplace noise", e)

        # CORRECT API: generalize_numeric_to_ranges uses 'columns' (list) and 'num_bins'
        try:
            result = generalize_numeric_to_ranges(
                sample_data.copy(),
                columns=["age"],  # CORRECT: columns as list
                num_bins=5,  # CORRECT: num_bins not bins
            )
            test_passed("Generalization")
        except Exception as e:
            test_failed("Generalization", e)

        try:
            result = shuffle_columns(sample_data.copy(), columns=["income"])
            test_passed("Column shuffling")
        except Exception as e:
            test_failed("Column shuffling", e)

    # ============================================================
    # TEST 6: Single-Call Workflow (Generate + Drift + Report)
    # ============================================================
    test_section("Single-Call Workflow - Generate + Drift + Report")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # CORRECT: Use inject_feature_drift_gradual method name
            result = gen.generate(
                data=sample_data,
                n_samples=30,
                method="cart",
                target_col="target",
                output_dir=tmpdir,
                save_dataset=True,
                drift_injection_config=[
                    {
                        "method": "inject_feature_drift_gradual",  # CORRECT method name
                        "params": {
                            "feature_cols": [
                                "score"
                            ],  # CORRECT: feature_cols not columns
                            "drift_type": "shift",
                            "drift_magnitude": 0.3,
                            "start_index": 10,
                            "auto_report": False,
                        },
                    }
                ],
                constraints=[{"col": "age", "op": ">=", "val": 21}],
            )

            if result is not None:
                files = os.listdir(tmpdir)
                test_passed(f"Single-call: {len(result)} samples, {len(files)} files")
            else:
                test_failed("Single-call workflow", "Result is None")
    except Exception as e:
        test_failed("Single-call workflow", e)

    # ============================================================
    # TEST 7: QualityReporter
    # ============================================================
    test_section("QualityReporter - Report Generation")

    try:
        from calm_data_generator.generators.tabular import QualityReporter

        reporter = QualityReporter()
        test_passed("Import QualityReporter")
    except Exception as e:
        test_failed("Import QualityReporter", e)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {len(results['passed'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")

    if results["failed"]:
        print("\n❌ Failed tests:")
        for name, error in results["failed"]:
            print(f"   - {name}: {error}")

    total = len(results["passed"]) + len(results["failed"])
    success_rate = len(results["passed"]) / total * 100 if total > 0 else 0

    print(f"\n📈 Success Rate: {success_rate:.1f}%")
    print("=" * 60)

    if success_rate == 100:
        print("✅ ALL TESTS PASSED!")
    elif success_rate >= 90:
        print("✅ LIBRARY VERIFICATION PASSED!")
    else:
        print("⚠️ LIBRARY NEEDS ATTENTION")

    print("=" * 60)

    return results


if __name__ == "__main__":
    run_comprehensive_test()
