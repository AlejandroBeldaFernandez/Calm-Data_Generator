# `ClinicalDataGenerator` Documentation

The `ClinicalDataGenerator` is a powerful tool for creating synthetic clinical datasets, including demographic information, gene expression data (RNA-Seq and Microarray), and protein expression data. It is designed to be flexible, allowing for the generation of complex datasets with various types of correlations and disease effects.

## Installation

The `ClinicalDataGenerator` is part of the `calmops` package and does not require separate installation. Ensure that the `calmops` package and its dependencies are correctly installed in your environment.

## Basic Usage

To use the `ClinicalDataGenerator`, you first need to import it and create an instance:

```python
from calmops.data_generators.Clinic.Clinic import ClinicalDataGenerator

# Initialize the generator with a seed for reproducibility
generator = ClinicalDataGenerator(seed=42)
```

## Main Features

### 1. Unified Generation (Recommended)

The `generate` method creates all data types (Demographics, Genes, Proteins) in one go, handling correlations automatically.

```python
from calmops.data_generators.Clinic.Clinic import ClinicalDataGenerator

generator = ClinicalDataGenerator(seed=42)

# Generate everything
datasets = generator.generate(
    n_patients=100,
    n_genes=200,
    n_proteins=50,
    # Optional: Define target variable
    target_variable_config={"weights": {"Age": 0.5}, "name": "diagnosis"}
)

demo_df = datasets["demographics"]
genes_df = datasets["genes"]
print(demo_df.head())
```

### 2. Component Generation (Advanced)

If you need granular control, you can call individual methods:

#### Generating Demographic Data
```python
demographic_df, raw_demographic_data = generator.generate_demographic_data(
    n_samples=150,
    control_disease_ratio=0.6
)
```

### 2. Generating Gene Expression Data

You can generate two types of gene expression data: `RNA-Seq` and `Microarray`. The `generate_gene_data` method allows you to specify disease effects to simulate biological conditions.

```python
# Example for RNA-Seq data with disease effects
gene_effects_config = {
    'effects': {
        'up_regulated': {'indices': list(range(10)), 'effect_type': 'fold_change', 'effect_value': 2.5},
        'down_regulated': {'indices': list(range(10, 20)), 'effect_type': 'fold_change', 'effect_value': 0.5}
    },
    'patient_subgroups': [
        {'name': 'Subgroup1', 'percentage': 0.5, 'apply_effects': ['up_regulated']},
        {'name': 'Subgroup2', 'remainder': True, 'apply_effects': ['down_regulated']}
    ]
}

rna_seq_df = generator.generate_gene_data(
    n_genes=50,
    gene_type="RNA-Seq",
    demographic_df=demographic_df,
    demographic_id_col=demographic_df.index.name,
    disease_effects_config=gene_effects_config
)

print(rna_seq_df.head())
```

### 3. Generating Protein Expression Data

Similar to gene data, you can generate protein expression data with specified disease effects.

```python
protein_effects_config = [
    {'name': 'Protein_Effect_1', 'indices': list(range(5)), 'effect_type': 'additive_shift', 'effect_value': 1.5}
]

protein_df = generator.generate_protein_data(
    n_proteins=30,
    demographic_df=demographic_df,
    demographic_id_col=demographic_df.index.name,
    disease_effects_config=protein_effects_config
)

print(protein_df.head())
```

### 4. Injecting Data Drift

The `ClinicalDataGenerator` provides methods to simulate data drift, which is crucial for testing the robustness of machine learning models over time.

#### Group Transition Drift

This method simulates patients transitioning from one group to another (e.g., 'Control' to 'Disease').

```python
# Concatenate omics data
omics_df = pd.concat([rna_seq_df, protein_df], axis=1)

transition_drift_config = {
    'transition_type': 'control_to_disease',
    'selection_criteria': {'percentage': 0.2},
    'omics_type': 'both',
    'disease_gene_indices': list(range(10)),
    'disease_protein_indices': list(range(5)),
    'disease_effect_type': 'fold_change',
    'disease_effect_value': 2.0
}

drifted_demographic_df, drifted_omics_df = generator.inject_drift_group_transition(
    demographic_df=demographic_df,
    omics_data_df=omics_df,
    n_genes_total=50,
    n_proteins_total=30,
    gene_type="RNA-Seq",
    **transition_drift_config
)

print(f"Number of transitions: {(drifted_demographic_df['Grupo'] != demographic_df['Grupo']).sum()}")
```

### 5. Generating Target Variable (Diagnosis)

The `generate_target_variable` method allows you to create a target variable (e.g., diagnosis) based on a linear combination of demographic and omics features.

```python
weights = {
    "Age": 0.3,
    "Sex": 0.1,
    "Gene_0": 0.05
}

diagnosis = generator.generate_target_variable(
    demographic_df=raw_demographic_data,
    omics_dfs=genes_df,
    weights=weights,
    binary_threshold=0.0 # Optional: binarize the output
)
```

## Advanced Usage: Custom Correlations

To generate complex correlation structures between demographic variables and omics groups, you can construct a block correlation matrix.

The `tutorial.py` script includes a helper function `build_correlation_matrix` that simplifies this process. It supports:
- **Internal Correlations:** Correlation between features within the same group. Can be a fixed value (e.g., `0.5`) or a range (e.g., `(0.3, 0.6)`) to sample from.
- **Demographic Correlations:** Correlation between a group and a demographic variable (e.g., Group A correlated with Age).

## Example Script / Tutorial

For a complete, step-by-step guide on how to use the `ClinicalDataGenerator` to create a full dataset with complex correlations and a target variable, please refer to the `tutorial.py` script located in this directory.

To run the tutorial:
```bash
python tutorial.py
```
This will create a directory named `tutorial_output` containing the generated datasets (`demographics.csv`, `genes.csv`).
