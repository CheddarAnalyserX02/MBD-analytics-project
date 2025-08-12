import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
INPUT_PATH = os.path.join("data", "bank_features.csv")
EDA_OUTPUT_DIR = os.path.join("outputs", "eda")

# Ensure output directory exists
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

def run_eda(input_path=INPUT_PATH):
    # 1. Load data
    df = pd.read_csv(input_path)
    print(f"Loaded dataset: {df.shape}")

    # 2. Summary statistics
    print("\n===== DATA INFO =====")
    print(df.info())

    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe(include="all"))

    # Save summary to CSV
    summary_stats = df.describe(include="all")
    summary_stats.to_csv(os.path.join(EDA_OUTPUT_DIR, "summary_statistics.csv"))

    # 3. Distribution of Age
    if "age" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["age"], bins=20, kde=True)
        plt.title("Age Distribution")
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, "age_distribution.png"))
        plt.close()

    # 4. Target variable balance
    if "y" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="y", data=df)
        plt.title("Target Variable Distribution")
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, "target_distribution.png"))
        plt.close()

    # 5. Campaign success ratio vs Target
    if "campaign_success_ratio" in df.columns and "y" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="y", y="campaign_success_ratio", data=df)
        plt.title("Campaign Success Ratio by Target")
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, "campaign_success_vs_target.png"))
        plt.close()

    # 6. Correlation heatmap
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()

    print(f"✅ EDA completed. Results saved to {EDA_OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda()
