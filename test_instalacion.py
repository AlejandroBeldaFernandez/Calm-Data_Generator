"""
Test de instalación de CALM-Data-Generator
==========================================
Ejecuta: python test_instalacion.py
"""

import sys

print("=" * 50)
print("🧪 TEST DE INSTALACIÓN - CALM-Data-Generator")
print("=" * 50)

tests_passed = 0
tests_failed = 0

# 1. Imports básicos
print("\n1️⃣ Imports básicos...")
try:
    from calm_data_generator import (
        RealGenerator,
        ClinicalDataGenerator,
        DriftInjector,
        ScenarioInjector,
        QualityReporter,
    )

    print("   ✅ Módulos principales")
    tests_passed += 1
except ImportError as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 2. Anonymizer
print("\n2️⃣ Anonymizer...")
try:
    from calm_data_generator.anonymizer import (
        pseudonymize_columns,
        add_laplace_noise,
        shuffle_columns,
    )

    print("   ✅ Funciones de anonimización")
    tests_passed += 1
except ImportError as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# 3. Deep Learning (SDV)
print("\n3️⃣ Deep Learning (SDV)...")
try:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

    print("   ✅ SDV disponible")
    tests_passed += 1
except ImportError as e:
    print(f"   ⚠️ SDV no instalado: {e}")
    tests_failed += 1

# 4. PyTorch
print("\n4️⃣ PyTorch...")
try:
    import torch

    print(f"   ✅ PyTorch {torch.__version__}")
    tests_passed += 1
except ImportError as e:
    print(f"   ⚠️ PyTorch no instalado")
    tests_failed += 1

# 5. Test funcional rápido
print("\n5️⃣ Test funcional (CART)...")
try:
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        {
            "a": np.random.randn(30),
            "b": np.random.randint(0, 100, 30),
            "target": np.random.choice([0, 1], 30),
        }
    )

    gen = RealGenerator()
    result = gen.generate(df, 10, method="cart", target_col="target")

    if result is not None and len(result) > 0:
        print(f"   ✅ CART funciona: {len(result)} muestras")
        tests_passed += 1
    else:
        print("   ❌ CART devolvió None")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ Error: {e}")
    tests_failed += 1

# Resumen
print("\n" + "=" * 50)
print(f"📊 RESUMEN: {tests_passed} pasados, {tests_failed} fallidos")
if tests_failed == 0:
    print("✅ ¡INSTALACIÓN CORRECTA!")
else:
    print("⚠️ Hay módulos que no funcionan")
print("=" * 50)

sys.exit(0 if tests_failed == 0 else 1)
