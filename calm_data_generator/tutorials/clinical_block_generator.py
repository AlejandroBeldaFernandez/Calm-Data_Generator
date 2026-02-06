"""
Tutorial 8: ClinicalBlockGenerator - Multi-Center Clinical Studies
==================================================================

This tutorial demonstrates how to use ClinicalDataGeneratorBlock to simulate data
from multiple clinical sites (blocks) or time periods, potentially with different
underlying distributions (simulating center effects or drift).
"""

import pandas as pd
import shutil
import os

try:
    from river import synth
except ImportError:
    from river.datasets import synth
from calm_data_generator.generators.clinical.ClinicGeneratorBlock import (
    ClinicalDataGeneratorBlock,
)

# Setup output directory
OUTPUT_DIR = "tutorial_output/08_clinic_block"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

gen = ClinicalDataGeneratorBlock()

print("\n--- Generating Multi-Center Clinical Data ---")

# Scenario: Simulating data from 3 different hospitals (blocks)
# We use River generators to seed the randomness/concept for each block
# If we wanted "center effects", we could use different seeds or generator types

# Hospital A (Seed 42)
gen_A = synth.Agrawal(classification_function=0, seed=42)
# Hospital B (Seed 100) - Different patterns
gen_B = synth.Agrawal(classification_function=0, seed=100)
# Hospital C (Seed 999) - Different patterns
gen_C = synth.Agrawal(classification_function=0, seed=999)


full_path = gen.generate(
    output_dir=OUTPUT_DIR,
    filename="multi_center_study.csv",
    n_blocks=3,
    total_samples=150,  # 50 patients per hospital
    n_samples_block=[50, 50, 50],
    generators=[gen_A, gen_B, gen_C],
    target_col="diagnosis",
    date_start="2024-01-01",
    date_step={
        "months": 1
    },  # Each block is separated by a month (or could be simultaneous)
    generate_report=True,  # Generate aggregated clinical report
)

print(f"Generated multi-center clinical data at: {full_path}")
df = pd.read_csv(full_path)

print("Columns generated:", df.columns.tolist()[:10], "...")
print("\nPatient counts per Hospital (Block):")
print(df["block"].value_counts())

print("\nSample records:")
print(df[["Patient_ID", "Age", "Sex", "diagnosis", "block"]].head())

print(f"\n✅ Tutorial completed! Outputs saved to {OUTPUT_DIR}")
