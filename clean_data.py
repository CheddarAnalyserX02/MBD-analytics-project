import pandas as pd
import os

# Paths
DATA_PATH = os.path.join("data", "bank-additional-full.csv")
OUTPUT_PATH = os.path.join("data", "bank_clean.csv")

def clean_bank_data(input_path=DATA_PATH, output_path=OUTPUT_PATH):
    # 1. Load data
    df = pd.read_csv(input_path, sep=';')
    print(f"Initial shape: {df.shape}")

    # 2. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 3. Handle missing values (in this dataset, they are marked as 'unknown')
    df.replace("unknown", pd.NA, inplace=True)

    # Option A: Drop rows with missing values
    df.dropna(inplace=True)

    # 4. Convert data types
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)

    # 5. Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 6. Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    clean_bank_data()
