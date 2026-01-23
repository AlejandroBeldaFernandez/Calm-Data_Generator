"""
Tutorial 7: StreamBlockGenerator - Simulating Concept Drift
===========================================================

This tutorial demonstrates how to use SyntheticBlockGenerator (StreamBlockGenerator)
to create datasets with scheduled concept drift by chaining multiple data stream generators.
"""

import pandas as pd
import shutil
import os
from river import synth
from calm_data_generator.generators.stream.StreamBlockGenerator import (
    SyntheticBlockGenerator,
)

# Setup output directory
OUTPUT_DIR = "tutorial_output/07_stream_block"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

gen = SyntheticBlockGenerator()

# ============================================================
# 1. Simple Interface (String-based)
# ============================================================
print("\n--- 1. Generating Blocks with SEA Generator (Simple Interface) ---")

# SEA generator with different thresholds/concepts for each block
final_path = gen.generate_blocks_simple(
    output_dir=OUTPUT_DIR,
    filename="sea_blocks_simple.csv",
    n_blocks=3,
    total_samples=3000,
    methods="sea",
    method_params=[
        {"variant": 0, "noise": 0.0},  # Block 1: Concept 0
        {"variant": 1, "noise": 0.05},  # Block 2: Concept 1 (Drift!) + Noise
        {"variant": 2, "noise": 0.1},  # Block 3: Concept 2 (Drift!) + More Noise
    ],
    generate_report=True,  # Generate comparison report
)

print(f"Generated simplified SEA dataset at: {final_path}")
df_simple = pd.read_csv(final_path)
print("Block counts:")
print(df_simple["block"].value_counts())


# ============================================================
# 2. Advanced Interface (River Objects)
# ============================================================
print("\n--- 2. Generating Blocks with Agrawal (Manual Interface) ---")

# We can manually instantiate River generators for fine-grained control
# Agrawal generator with changing classification functions
gen1 = synth.Agrawal(classification_function=0, seed=42)
gen2 = synth.Agrawal(classification_function=1, seed=42)  # Abrupt drift
gen3 = synth.Agrawal(classification_function=2, seed=42)  # Another abrupt drift

full_path_manual = gen.generate(
    output_dir=OUTPUT_DIR,
    filename="agrawal_manual_drift.csv",
    n_blocks=3,
    total_samples=1500,
    n_samples_block=[500, 500, 500],
    generators=[gen1, gen2, gen3],
    date_start="2024-01-01",
    date_step={"days": 30},
    generate_report=False,
)

print(f"Generated manual Agrawal dataset at: {full_path_manual}")
df_manual = pd.read_csv(full_path_manual)
print(df_manual.head())

print(f"\n✅ Tutorial completed! Outputs saved to {OUTPUT_DIR}")
