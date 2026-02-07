import pandas as pd
import numpy as np
import traceback
import sys

# Force output to stdout
sys.stdout.reconfigure(encoding="utf-8")

print(
    "1. Creating LARGER sample data with combinatorial condition names and 50 genes...",
    flush=True,
)
np.random.seed(42)
n_samples = 300

# Use real gene names for GO compatibility, plus extra generic ones
gene_names = [f"GENE_{i}" for i in range(50)]
gene_names[:10] = [
    "TP53",
    "EGFR",
    "TNF",
    "MAPK1",
    "GAPDH",
    "ACTB",
    "MYC",
    "BRCA1",
    "IL6",
    "INS",
]

data = pd.DataFrame(np.random.randn(n_samples, 50), columns=gene_names)
# Add conditions:
conditions = (
    ["ctrl"] * 100
    + ["TP53+ctrl"] * 50
    + ["EGFR+ctrl"] * 50
    + ["TNF+ctrl"] * 50
    + ["MYC+ctrl"] * 50
)
data["condition"] = conditions

print(f"   Data shape: {data.shape}")
print("   Conditions:", data["condition"].value_counts().to_dict())

print("2. Importing RealGenerator...", flush=True)
try:
    from calm_data_generator.generators.tabular import RealGenerator

    print("   RealGenerator imported.", flush=True)
except Exception as e:
    print(f"❌ Failed to import RealGenerator: {e}", flush=True)
    sys.exit(1)

print("3. Initializing generator...", flush=True)
gen = RealGenerator(auto_report=False)

print("4. Testing GEARS synthesis...", flush=True)
try:
    # We use target_col='condition' so it gets preserved
    synthetic = gen.generate(
        data=data,
        n_samples=50,
        target_col="condition",
        method="gears",
        perturbations=["TP53+ctrl", "TNF+ctrl"],
        epochs=1,
        device="cpu",
    )
    if synthetic is not None:
        print(f"✅ GEARS synthesis successful!", flush=True)
        print(f"   Generated {len(synthetic)} samples", flush=True)
        print(f"   Columns: {list(synthetic.columns[:5])}", flush=True)
    else:
        print("⚠️ GEARS returned None", flush=True)
except Exception as e:
    print(f"❌ GEARS synthesis failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
