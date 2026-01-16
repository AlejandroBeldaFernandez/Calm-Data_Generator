import pandas as pd
import numpy as np
import hashlib

def pseudonymize_columns(df: pd.DataFrame, columns: list, salt: str = None) -> pd.DataFrame:
    """
    Pseudonymizes specified columns in a DataFrame using SHA256 hashing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to pseudonymize.
        salt (str, optional): An optional salt to add to the hashing process for security.
                              It is highly recommended to use a salt.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns pseudonymized.
    """
    df_copy = df.copy()
    # Use an empty string if salt is None for consistent hashing behavior
    salt = salt or ''
    
    for col in columns:
        if col in df_copy.columns:
            # Ensure the data is string and encoded to bytes for hashing
            df_copy[col] = df_copy[col].astype(str).apply(
                lambda x: hashlib.sha256((x + salt).encode()).hexdigest()
            )
    return df_copy

def add_laplace_noise(df: pd.DataFrame, columns: list, epsilon: float = 1.0) -> pd.DataFrame:
    """
    Applies Laplace noise to specified numeric columns for differential privacy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of numeric column names to add noise to.
        epsilon (float): The privacy budget (a smaller value means more privacy and more noise).
                         Defaults to 1.0.

    Returns:
        pd.DataFrame: A new DataFrame with noise added to specified columns.
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            # For simplicity, sensitivity is assumed to be the range of the column.
            # In a real-world scenario, this should be determined more carefully.
            col_range = df_copy[col].max() - df_copy[col].min()
            
            if col_range > 0 and epsilon > 0:
                scale = col_range / epsilon
                noise = np.random.laplace(0, scale, len(df_copy))
                df_copy[col] = df_copy[col] + noise
    return df_copy

def generalize_numeric_to_ranges(df: pd.DataFrame, columns: list, num_bins: int = 5) -> pd.DataFrame:
    """
    Generalizes specified numeric columns by binning their values into ranges.
    This is a common technique for achieving k-anonymity, reducing re-identifiability.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of numeric column names to generalize.
        num_bins (int): The number of bins or ranges to create for each column.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns generalized into string-based ranges.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            # Use pandas.cut to segment the data into bins and convert them to string representations
            df_copy[col] = pd.cut(df_copy[col], bins=num_bins, include_lowest=True, duplicates='drop').astype(str)
    return df_copy

def generalize_categorical_by_mapping(df: pd.DataFrame, columns: list, mapping: dict) -> pd.DataFrame:
    """
    Generalizes specified categorical columns by applying a user-defined mapping.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of categorical column names to generalize.
        mapping (dict): A dictionary that defines how to map old values to new, generalized values.
                        Values not found in the mapping will be kept as they are.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns generalized according to the mapping.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            # Apply the mapping. Use fillna to keep original values that are not in the map.
            df_copy[col] = df_copy[col].map(mapping).fillna(df_copy[col])
    return df_copy

def shuffle_columns(df: pd.DataFrame, columns: list, random_state: int = None) -> pd.DataFrame:
    """
    Shuffles the values within specified columns independently.

    This method is useful for breaking correlations between columns for privacy,
    while preserving the distribution of each shuffled column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to shuffle.
        random_state (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns shuffled.
    """
    df_copy = df.copy()
    rng = np.random.default_rng(random_state)
    
    for col in columns:
        if col in df_copy.columns:
            # Shuffle the column values in place
            shuffled_values = df_copy[col].to_numpy()
            rng.shuffle(shuffled_values)
            df_copy[col] = shuffled_values
            
    return df_copy