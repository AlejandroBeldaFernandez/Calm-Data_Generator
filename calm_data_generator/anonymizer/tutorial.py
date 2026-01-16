import pandas as pd
from calm_data_generator.anonymizer.privacy import (
    pseudonymize_columns,
    add_laplace_noise,
    generalize_numeric_to_ranges,
    generalize_categorical_by_mapping,
    shuffle_columns,
)


def run_tutorial():
    print("=== Privacy Module Tutorial ===")

    # 1. Create Sample Data
    print("\n1. Creating Sample Data...")
    df = pd.DataFrame(
        {
            "user_id": ["User_001", "User_002", "User_003", "User_004", "User_005"],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [24, 35, 42, 29, 55],
            "salary": [52000, 65000, 72000, 48000, 95000],
            "city": ["New York", "Los Angeles", "Chicago", "New York", "San Francisco"],
        }
    )
    print("Original Data:\n", df)

    # 2. Pseudonymization
    print("\n2. Pseudonymizing 'user_id' and 'name'...")
    # Using a salt for added security
    df_pseudo = pseudonymize_columns(
        df, columns=["user_id", "name"], salt="my_secret_salt"
    )
    print(df_pseudo[["user_id", "name"]].head())

    # 3. Differential Privacy (Laplace Noise)
    print("\n3. Adding Laplace Noise to 'salary' (epsilon=0.1)...")
    # Lower epsilon = more noise = more privacy
    df_noise = add_laplace_noise(df_pseudo, columns=["salary"], epsilon=0.1)
    print("Original Salaries:", df["salary"].tolist())
    print("Noisy Salaries:   ", df_noise["salary"].astype(int).tolist())

    # 4. Generalization (Numeric)
    print("\n4. Generalizing 'age' into ranges...")
    df_gen_num = generalize_numeric_to_ranges(df_noise, columns=["age"], num_bins=3)
    print(df_gen_num[["age"]].head())

    # 5. Generalization (Categorical)
    print("\n5. Generalizing 'city' to regions...")
    city_mapping = {
        "New York": "East Coast",
        "Los Angeles": "West Coast",
        "Chicago": "Midwest",
        "San Francisco": "West Coast",
    }
    df_gen_cat = generalize_categorical_by_mapping(
        df_gen_num, columns=["city"], mapping=city_mapping
    )
    print(df_gen_cat[["city"]].head())

    # 6. Shuffling
    print("\n6. Shuffling 'salary' to break correlations...")
    # Note: This preserves the distribution of salaries but breaks the link to specific rows
    df_final = shuffle_columns(df_gen_cat, columns=["salary"], random_state=42)

    print("\n=== Final Anonymized Dataset ===")
    print(df_final)


if __name__ == "__main__":
    run_tutorial()
