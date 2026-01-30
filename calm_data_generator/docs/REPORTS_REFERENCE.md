# Reports Reference

The `calm_data_generator` library includes a suite of reporting tools designed to assess the quality, privacy, and characteristics of generated data.

## Quality Reporter (`Tabular`)
**Module:** `calm_data_generator.generators.tabular.QualityReporter`

Generates comprehensive reports comparing real and synthetic tabular data.

### `generate_comprehensive_report`
Generates a static report including:
- **SDV Quality Scores**: Overall and column-wise similarity metrics.
- **Privacy Assessment**: Distance to Closest Record (DCR) metrics.
- **Visualizations**: Histograms, density plots, PCA/UMAP projections.
- **Drift Analysis**: Visual comparison of feature distributions.

```python
reporter = QualityReporter(verbose=True)
reporter.generate_comprehensive_report(
    real_df=original_df,
    synthetic_df=synthetic_df,
    generator_name="MyGenerator",
    output_dir="./report_output"
)
```

    output_dir="./report_output"
)
```

## Discriminator Reporter (Adversarial Validation)
**Module:** `calm_data_generator.reports.DiscriminatorReporter`

This reporter trains a classifier model (Random Forest) to attempt to distinguish between real and synthetic data. It is used to detect drift or assess general fidelity.

### Key Metrics
- **Similarity Score (Indistinguishability)**: (0.0 - 1.0).
    - **Formula**: `1 - 2 * |AUC - 0.5|`
    - `1.0`: Indistinguishable data (AUC = 0.5). Excellent Quality.
    - `0.0`: Easily distinguishable data (AUC = 1.0 or 0.0). Drift detected or poor quality.
- **Confusion Score**: Ability of the data to "confuse" the discriminator (based on inverted Accuracy).
- **Explainability**:
    - **Feature Importance**: Which variables allowed the model to distinguish the data.
    - **SHAP Values**: Detailed explanation of the impact of each feature.

### Usage
This reporter is automatically integrated into `QualityReporter` if the optional parameter is activated:
```python
reporter.generate_comprehensive_report(
    ...,
    adversarial_validation=True  # Activate Discriminator
)
```

## Stream Reporter (`Stream`)
**Module:** `calm_data_generator.generators.stream.StreamReporter`

Designed for analyzing synthetic data streams without a direct "real" reference dataset (though it can compare against expectations).

### `generate_report`
Generates a report for a synthetic dataset:
- **Data Profiling**: YData Profiling integration.
- **Visualizations**: Density plots and dimensionality reduction.
- **Block-wise Analysis**: Can generate separate reports for each data block.

```python
reporter = StreamReporter()
reporter.generate_report(
    synthetic_df=stream_df,
    generator_name="StreamGen",
    output_dir="./stream_report"
)
```

## Privacy Reporter (`Anonymizer`)
**Module:** `calm_data_generator.anonymizer.PrivacyReporter`

Focuses specifically on privacy metrics and the trade-off between privacy and utility.

### `generate_privacy_report`
Generates an HTML report visualizing:
- **Uniqueness Change**: How anonymization affected record uniqueness.
- **Correlation Loss**: Changes in feature correlations.
- **Distribution Overlay**: Visual comparison of original vs. anonymized features.

```python
PrivacyReporter.generate_privacy_report(
    original_df=df,
    private_df=anonymized_df,
    output_dir="./privacy_report"
)
```

## Clinic Reporter (`Clinical`)
**Module:** `calm_data_generator.generators.clinical.ClinicReporter`

A specialized version of `StreamReporter` for clinical data. It inherits standard reporting capabilities but is tailored to handle clinical feature sets and may include domain-specific checks in the future.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```
