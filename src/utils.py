import pandas as pd
import joblib
import os

def load_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def save_csv(df: pd.DataFrame, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)

def print_df_info(df: pd.DataFrame, n_rows: int = 5):
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Head:")
    print(df.head(n_rows))
    print("Info:")
    print(df.info())
    print("Missing values counts:")
    print(df.isnull().sum())

