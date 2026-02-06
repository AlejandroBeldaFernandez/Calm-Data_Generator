# Referencia de Informes (Reports)

La biblioteca `calm_data_generator` incluye un conjunto de herramientas de informes diseñadas para evaluar la calidad, privacidad y características de los datos generados.

## Reporter de Calidad (`Tabular`)
**Módulo:** `calm_data_generator.generators.tabular.QualityReporter`

Genera informes completos comparando datos tabulares reales y sintéticos.

### `generate_comprehensive_report`
Genera un informe estático incluyendo:
- **Puntuaciones de Calidad Global**: Métricas de similitud generales y por columna.
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

## Reporter Discriminador (Adversarial Validation)
**Módulo:** `calm_data_generator.reports.DiscriminatorReporter`

Este reporter entrena un modelo clasificador (Random Forest) para intentar distinguir entre datos reales y sintéticos. Se utiliza para detectar drift o evaluar la fidelidad general.

### Métricas Clave
- **Similarity Score (Indistinguishability)**: (0.0 - 1.0).
    - **Fórmula**: `1 - 2 * |AUC - 0.5|`
    - `1.0`: Datos indistinguibles (AUC = 0.5). Excelente Calidad.
    - `0.0`: Datos fácilmente distinguibles (AUC = 1.0 o 0.0). Drift detectado o baja calidad.
- **Confusion Score**: Capacidad de los datos para "confundir" al discriminador (basado en Accuracy inversamente).
- **Explicabilidad**:
    - **Feature Importance**: Qué variables permitieron al modelo distinguir los datos.
    - **SHAP Values**: Explicación detallada del impacto de cada feature.

### Uso
Este reporter se integra automáticamente en `QualityReporter` si se activa el parámetro opcional:
```python
reporter.generate_comprehensive_report(
    ...,
    adversarial_validation=True  # Activar Discriminator
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


## Reporter Clínico (`Clinical`)
**Módulo:** `calm_data_generator.generators.clinical.ClinicReporter`

Una versión especializada de `StreamReporter` para datos clínicos. Hereda capacidades de reporte estándar pero está adaptado para manejar conjuntos de características clínicas y puede incluir verificaciones específicas de dominio en el futuro.

```python
reporter = ClinicReporter()
reporter.generate_report(...)
```

> [!NOTE]
> **Informes de Privacidad**: Las características de privacidad (métricas DCR) ahora están integradas en `QualityReporter`. Usa `privacy_check=True` al generar informes.

