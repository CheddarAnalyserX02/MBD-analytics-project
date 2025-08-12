import pandas as pd
import os

# Paths
INPUT_PATH = os.path.join("data", "bank_clean.csv")
OUTPUT_PATH = os.path.join("data", "bank_features.csv")

def engineer_features(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    # 1. Load cleaned data
    df = pd.read_csv(input_path)
    print(f"Loaded cleaned data: {df.shape}")

    # 2. Feature: Age groups
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[17, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "65+"]
        )

    # 3. Feature: Campaign effectiveness ratio (last contact success rate)
    if "campaign" in df.columns and "previous" in df.columns:
        df["campaign_success_ratio"] = df["previous"] / (df["campaign"] + 1)
        df["campaign_success_ratio"].fillna(0, inplace=True)

    # 4. Feature: Total contact count
    if "campaign" in df.columns and "previous" in df.columns:
        df["total_contacts"] = df["campaign"] + df["previous"]

    # 5. Feature: Call duration per campaign
    if "duration" in df.columns and "campaign" in df.columns:
        df["avg_call_duration"] = df["duration"] / (df["campaign"] + 1)

    # 6. One-hot encode age group
    if "age_group" in df.columns:
        df = pd.get_dummies(df, columns=["age_group"], drop_first=True)

    # 7. Save new dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path}")
    print(f"Final shape: {df.shape}")

if __name__ == "__main__":
    engineer_features()
