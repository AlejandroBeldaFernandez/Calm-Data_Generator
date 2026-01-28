# Referencia de Informes (Reports)

La biblioteca `calm_data_generator` incluye un conjunto de herramientas de informes diseñadas para evaluar la calidad, privacidad y características de los datos generados.

## Reporter de Calidad (`Tabular`)
**Módulo:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes completos comparando datos tabulares reales y sintéticos.

### `generate_comprehensive_report`
Genera un informe estático incluyendo:
- **Puntuaciones de Calidad SDV**: Métricas de similitud generales y por columna.
- **Evaluación de Privacidad**: Métricas de Distancia al Registro Más Cercano (DCR).
- **Visualizaciones**: Histogramas, gráficos de densidad, proyecciones PCA/UMAP.
- **Análisis de Drift**: Comparación visual de distribuciones de features.

```python
reporter = QualityReporter(verbose=True)
reporter.generate_comprehensive_report(
    real_df=original_df,
    synthetic_df=synthetic_df,
    generator_name="MyGenerator",
    output_dir="./report_output"
)
```

## Reporter de Stream (`Stream`)
**Módulo:** `calm_data_generator.generators.stream.StreamReporter`

Diseñado para analizar flujos de datos sintéticos sin un dataset de referencia "real" directo (aunque puede comparar contra expectativas).

### `generate_report`
Genera un informe para un dataset sintético:
- **Perfilado de Datos**: Integración con YData Profiling.
- **Visualizaciones**: Gráficos de densidad y reducción de dimensionalidad.
- **Análisis por Bloques**: Puede generar informes separados para cada bloque de datos.

```python
reporter = StreamReporter()
reporter.generate_report(
    synthetic_df=stream_df,
    generator_name="StreamGen",
    output_dir="./stream_report"
)
```

## Reporter de Privacidad (`Anonymizer`)
**Módulo:** `calm_data_generator.anonymizer.PrivacyReporter`

Se centra específicamente en métricas de privacidad y el balance entre privacidad y utilidad.

### `generate_privacy_report`
Genera un informe HTML visualizando:
- **Cambio de Unicidad**: Cómo afectó la anonimización a la unicidad de registros.
- **Pérdida de Correlación**: Cambios en correlaciones de features.
- **Superposición de Distribuciones**: Comparación visual de features originales vs. anonimizadas.

```python
PrivacyReporter.generate_privacy_report(
    original_df=df,
    private_df=anonymized_df,
    output_dir="./privacy_report"
)
```

## Reporter Clínico (`Clinical`)
**Módulo:** `calm_data_generator.generators.clinical.ClinicReporter`

Una versión especializada de `StreamReporter` para datos clínicos. Hereda capacidades de reporte estándar pero está adaptado para manejar conjuntos de características clínicas y puede incluir verificaciones específicas de dominio en el futuro.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```
