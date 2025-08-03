import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, KBinsDiscretizer

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def preprocess_and_engineer(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])


    for col in cat_cols:
        if set(df[col].str.lower().unique()) & {'yes', 'no'}:
            df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0}).fillna(df[col])


    if 'Monthly_Income' in df.columns and 'Loan_Amount' in df.columns:
        df['inc_loan_ratio'] = df['Monthly_Income'] / (df['Loan_Amount'] + 1)

    if 'Loan_Amount' in df.columns:
        df['Loan_Amount_log'] = np.log1p(df['Loan_Amount'])

    if 'Loan_Amount' in df.columns:
        kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        df['Loan_Amount_bin'] = kbins.fit_transform(df[['Loan_Amount']]).astype(int)

    if 'Monthly_Income' in df.columns and 'Total_Debt' in df.columns:
        df['debt_to_income'] = df['Total_Debt'] / (df['Monthly_Income'] + 1)

    low_card_cat = [col for col in cat_cols if df[col].nunique() <= 5 and col != 'Label']
    df = pd.get_dummies(df, columns=low_card_cat, drop_first=True)

    label_enc = LabelEncoder()
    for col in cat_cols:
        if col not in low_card_cat + ['Label']:
            df[col] = label_enc.fit_transform(df[col])

    for col in ['Loan_ID', 'customer_id']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)


    return df

if __name__ == "__main__":
    input_path = "data/Loan_Default.csv"
    output_path = "data/Loan_Default_Processed.csv"

    print(f"Loading data from: {input_path}")
    df = load_data(input_path)
    print("Initial shape:", df.shape)
    df = preprocess_and_engineer(df)
    print("Final shape after feature engineering:", df.shape)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")