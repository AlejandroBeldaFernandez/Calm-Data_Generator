# ClinicalDataGenerator API Reference

The `ClinicalDataGenerator` class provides a comprehensive suite of tools for generating synthetic clinical datasets, including demographic, gene expression (RNA-Seq and Microarray), and protein expression data. It supports complex correlation structures, disease effects, and various types of data drift.

## Class Initialization

### `__init__(self, seed=42)`
Initializes the generator with a random seed for reproducibility.

- **Parameters:**
  - `seed` (int): Random seed. Default is 42.

---

## Main Generation Method

### `generate`
Unified entry point for generating clinical datasets (Demographics + Omics).

```python
def generate(
    self,
    n_patients: int = 100,
    n_genes: int = 200,
    n_proteins: int = 50,
    date_config: Optional[DateConfig] = None,
    output_dir: Optional[str] = None,
    save_dataset: bool = False,
    # Kwargs for sub-generators:
    # control_disease_ratio, custom_demographic_columns,
    # gene_type, demographic_gene_correlations, gene_correlations,
    # target_variable_config, drift_injection_config, etc.
    **kwargs
) -> Dict[str, pd.DataFrame]
```

- **Returns:** Dictionary with keys `['demographics', 'genes', 'proteins']`.

---

## Component Methods (Advanced)

### `generate_demographic_data`
Generates synthetic demographic data for a cohort of patients.

```python
def generate_demographic_data(
    self,
    n_samples: int,
    control_disease_ratio: float = 0.5,
    demographic_correlations: np.ndarray = None,
    custom_demographic_columns: dict = None,
    date_column_name: str = None,
    date_value: str = None,
    class_assignment_function: callable = None
) -> (pd.DataFrame, pd.DataFrame)
```

- **Parameters:**
  - `n_samples` (int): Number of patients to generate.
  - `control_disease_ratio` (float): Ratio of control patients (0.0 to 1.0). Default 0.5.
  - `demographic_correlations` (np.ndarray): Correlation matrix for demographic features.
  - `custom_demographic_columns` (dict): Dictionary defining custom columns and their distributions (scipy.stats objects or dicts).
  - `date_column_name` (str): Name for an optional date column.
  - `date_value` (str): Value for the date column.
  - `class_assignment_function` (callable): Custom function to assign disease subgroups.

- **Returns:**
  - `df_temp` (pd.DataFrame): The main demographic DataFrame with categorical values.
  - `raw_demographic_data` (pd.DataFrame): DataFrame with raw numerical values (useful for correlations).

### `generate_gene_data`
Generates synthetic gene expression data (RNA-Seq or Microarray).

```python
def generate_gene_data(
    self,
    n_genes: int,
    gene_type: str,
    demographic_df: pd.DataFrame = None,
    demographic_id_col: str = None,
    raw_demographic_data: pd.DataFrame = None,
    gene_correlations: np.ndarray = None,
    demographic_gene_correlations: np.ndarray = None,
    disease_effects_config: dict = None,
    subgroup_col: str = None,
    gene_mean_log_center: float = np.log(80),
    gene_mean_loc_center: float = 7.0,
    control_disease_ratio: float = 0.5,
    custom_gene_parameters: dict = None,
    n_samples: int = 100
) -> pd.DataFrame
```

- **Parameters:**
  - `n_genes` (int): Number of genes to generate.
  - `gene_type` (str): "RNA-Seq" or "Microarray".
  - `demographic_df` (pd.DataFrame): DataFrame containing patient demographics.
  - `disease_effects_config` (dict): Configuration for applying disease effects to specific genes/subgroups.
  - `subgroup_col` (str): Column in `demographic_df` to use for subgroup-based effects.

- **Returns:**
  - `df_genes` (pd.DataFrame): Gene expression matrix (rows=patients, cols=genes).

### `generate_protein_data`
Generates synthetic protein expression data.

```python
def generate_protein_data(
    self,
    n_proteins: int,
    demographic_df: pd.DataFrame = None,
    demographic_id_col: str = None,
    raw_demographic_data: pd.DataFrame = None,
    protein_correlations: np.ndarray = None,
    demographic_protein_correlations: np.ndarray = None,
    disease_effects_config: list = None,
    control_disease_ratio: float = 0.5,
    custom_protein_parameters: dict = None,
    n_samples: int = 100
) -> pd.DataFrame
```

- **Returns:**
  - `df_proteins` (pd.DataFrame): Protein expression matrix.

### `generate_target_variable`
Generates a target variable (e.g., diagnosis) based on a linear combination of features.

```python
def generate_target_variable(
    self,
    demographic_df: pd.DataFrame,
    omics_dfs: list[pd.DataFrame] | pd.DataFrame,
    weights: dict,
    noise_std: float = 0.1,
    binary_threshold: float = None
) -> pd.Series
```

- **Parameters:**
  - `weights` (dict): Mapping of column names (or regex) to their weights.
  - `binary_threshold` (float): If set, binarizes the output (0 or 1).

---

## Drift Injection Methods

### `inject_drift_group_transition`
Simulates drift by transitioning patients between 'Control' and 'Disease' groups and regenerating their omics data.

```python
def inject_drift_group_transition(
    self,
    demographic_df: pd.DataFrame,
    omics_data_df: pd.DataFrame,
    transition_type: str,
    selection_criteria: dict,
    omics_type: str,
    gene_type: str = None,
    disease_gene_indices: list = None,
    disease_protein_indices: list = None,
    disease_effect_type: str = None,
    disease_effect_value: float = None,
    n_genes_total: int = None,
    n_proteins_total: int = None
) -> (pd.DataFrame, pd.DataFrame)
```

- **Parameters:**
  - `transition_type` (str): 'control_to_disease', 'disease_to_control', or 'bidirectional'.
  - `selection_criteria` (dict): How to select patients (e.g., `{'percentage': 0.1}`).

### `inject_drift_correlated_modules`
Injects drift by modifying the correlation structure of specific omics modules.

```python
def inject_drift_correlated_modules(
    self,
    omics_data_df: pd.DataFrame,
    module_indices: list,
    new_correlation_matrix: np.ndarray = None,
    add_indices: list = None,
    remove_indices: list = None,
    omics_type: str = "genes",
    gene_type: str = None
) -> pd.DataFrame
```

### `generate_additional_time_step_data`
Generates a new batch of data for a subsequent time step, optionally injecting drift.

```python
def generate_additional_time_step_data(
    self,
    n_samples: int,
    date_value: str,
    omics_to_generate: list,
    n_genes: int = 0,
    n_proteins: int = 0,
    gene_type: str = None,
    parameter_drift_config: dict = None,
    transition_drift_config: dict = None,
    module_drift_config: dict = None,
    **kwargs
) -> (pd.DataFrame, pd.DataFrame)
```
