from calm_data_generator.generators.clinical import ClinicalDataGenerator
from calm_data_generator.generators.configs import DateConfig

gen = ClinicalDataGenerator(seed=42, auto_report=False, minimal_report=True)

biomarker_config = {
    "target_type": "gene",
    "index": [0, 5, 12],
    "effect_type": "fold_change",
    "effect_value": 2.0,
    "group": "Disease",
}

data = gen.generate(
    n_samples=100,
    n_genes=500,
    n_proteins=0,
    control_disease_ratio=0.5,
    date_config=DateConfig(start_date="2024-01-01"),
    disease_effects_config=[biomarker_config],
)

print("✓ Test passed! Generated data with keys:", list(data.keys()))
print("✓ Demographics shape:", data["demographics"].shape)
print("✓ Genes shape:", data["genes"].shape)
print("\n✓ Both errors fixed:")
print("  1. 'index' key is now accepted (not 'indices')")
print("  2. 'name' key is no longer required")
