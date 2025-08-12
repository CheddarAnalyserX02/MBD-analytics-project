# explore_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/bank-additional-full.csv", sep=';')

# ----- BASIC INFO -----
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# ----- QUICK SUMMARY -----
print("\nNumerical columns summary:")
print(df.describe())

print("\nCategorical columns summary:")
print(df.describe(include=['object']))

# ----- VISUAL EDA -----
# 1. Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='y', palette='pastel')
plt.title("Target Variable Distribution")
plt.savefig("plots/target_distribution.png")
plt.close()

# 2. Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.savefig("plots/age_distribution.png")
plt.close()

# 3. Job type vs subscription
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='job', hue='y', palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Job Type vs Subscription")
plt.savefig("plots/job_vs_subscription.png")
plt.close()

print("\nEDA plots saved in the 'plots/' folder.")

