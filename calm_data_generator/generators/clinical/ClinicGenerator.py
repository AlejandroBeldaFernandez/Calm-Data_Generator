import pandas as pd
from calm_data_generator.generators.stream.StreamGenerator import StreamGenerator


class ClinicalDataGenerator(StreamGenerator):
    """
    Specialized Synthetic Generator for Clinical Data.
    Inherits from StreamGenerator but adds domain-specific logic, such as ensuring
    feature names are clinically relevant (e.g., BloodPressure, HeartRate).
    """

    CLINICAL_FEATURE_MAP = {
        # String based (StreamGenerator fallback)
        "x0": "Systolic_BP",
        "x1": "Diastolic_BP",
        "x2": "Heart_Rate",
        "x3": "Body_Temp",
        "x4": "SpO2",
        "x5": "Respiratory_Rate",
        "x6": "Glucose",
        "x7": "Cholesterol",
        "x8": "BMI",
        "x9": "Age",
        # Integer based (River default keys or simple enumeration)
        0: "Systolic_BP",
        "0": "Systolic_BP",
        1: "Diastolic_BP",
        "1": "Diastolic_BP",
        2: "Heart_Rate",
        "2": "Heart_Rate",
        3: "Body_Temp",
        "3": "Body_Temp",
        4: "SpO2",
        "4": "SpO2",
        5: "Respiratory_Rate",
        "5": "Respiratory_Rate",
        6: "Glucose",
        "6": "Glucose",
        7: "Cholesterol",
        "7": "Cholesterol",
        8: "BMI",
        "8": "BMI",
        9: "Age",
        "9": "Age",
    }

    def _generate_internal(self, **kwargs) -> pd.DataFrame:
        """
        Overrides the internal generation to apply clinical feature mapping.
        """
        # Generate generic data using parent method
        # We temporarily disable report generation in the parent to avoid reporting on "x0", "x1" etc.
        original_gen_report = kwargs.get("generate_report", True)
        kwargs["generate_report"] = False

        df = super()._generate_internal(**kwargs)

        # Apply Mapping
        # Use available mapping for as many columns as present
        mapping = {
            k: v for k, v in self.CLINICAL_FEATURE_MAP.items() if k in df.columns
        }
        if mapping:
            df.rename(columns=mapping, inplace=True)

        # Restore report generation if it was requested
        if original_gen_report:
            # We must trigger report generation manually here because we skipped it in super()
            # to wait for renaming.
            out_dir = kwargs.get("output_dir") or self.DEFAULT_OUTPUT_DIR

            # Reconstruct variables needed for reporting
            report_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k != "save_dataset" and k != "generate_report"
            }
            # Ensure the report gets the actual generator instance
            report_kwargs["generator_instance"] = (
                kwargs.get("metadata_generator_instance")
                or kwargs["generator_instance"]
            )

            self._save_report_json(df=df, output_dir=out_dir, **report_kwargs)

        return df
