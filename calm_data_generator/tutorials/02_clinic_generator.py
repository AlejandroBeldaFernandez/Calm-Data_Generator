"""
Tutorial 2: ClinicalDataGenerator - Clinical/Medical Data Generation
===============================================================

This tutorial demonstrates how to generate realistic clinical data
including demographics, genes, proteins, and longitudinal studies.
"""

from calm_data_generator import ClinicalDataGenerator
from calm_data_generator.generators.configs import DateConfig

# ============================================================
# 1. Basic Clinical Data Generation
# ============================================================

gen = ClinicalDataGenerator()

# Generate demographics + omics data
result = gen.generate(
    n_samples=50,
    n_genes=100,
    n_proteins=50,
    date_config=DateConfig(start_date="2024-01-01"),
)

print("Generated datasets:")
for key, df in result.items():
    if df is not None:
        print(f"  {key}: {df.shape}")

# Access demographics
demographics = result.get("demographics")
print("\nDemographics columns:", demographics.columns.tolist()[:10])

# ============================================================
# 2. Clinical Constraints
# ============================================================

# Generate with clinical constraints (e.g., age limits)
constraints = [
    {"col": "Age", "op": ">=", "val": 18},
    {"col": "Age", "op": "<=", "val": 85},
]

result_constrained = gen.generate(n_samples=30, n_genes=50, constraints=constraints)

demo = result_constrained.get("demographics")
if demo is not None:
    print(f"\nConstrained Age range: {demo['Age'].min():.0f} - {demo['Age'].max():.0f}")

# ============================================================
# 3. Longitudinal Data (Multi-Visit Studies)
# ============================================================

# Generate longitudinal data with patient evolution
longitudinal_config = {
    "n_visits": 4,  # 4 visits per patient
    "time_step_days": 90,  # 90 days between visits
    "evolution_config": {
        "features": ["Age", "Propensity"],  # Features that evolve
        "trend": 0.02,  # 2% trend per visit
        "noise": 0.01,  # 1% random noise
    },
}

result_longitudinal = gen.generate_longitudinal_data(
    n_samples=20,
    longitudinal_config=longitudinal_config,
    date_config=DateConfig(start_date="2024-01-01"),
)

long_df = result_longitudinal.get("longitudinal")
print(f"\nLongitudinal data shape: {long_df.shape}")
print(f"Patients: {long_df['Patient_ID'].nunique()}")
print(f"Visits per patient: {long_df.groupby('Patient_ID').size().mean():.1f}")

# Show visits for one patient
patient_0 = long_df[long_df["Patient_ID"] == "P_0"]
print("\nPatient P_0 visits:")
print(patient_0[["Patient_ID", "Visit_ID", "Days_Since_Start", "Age"]].head())

# ============================================================
# 4. Gene and Protein Data
# ============================================================

genes = result.get("genes")
proteins = result.get("proteins")

if genes is not None:
    print(f"\nGene expression data: {genes.shape}")
    print(f"Gene columns (sample): {genes.columns.tolist()[:5]}")

if proteins is not None:
    print(f"Protein data: {proteins.shape}")

print("\n✅ Clinical data tutorial completed!")
