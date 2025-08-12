import pandas as pd

# Load dataset
df = pd.read_csv("data/bank-additional-full.csv", sep=';')

# Basic info
print("Shape:", df.shape)
print(df.head())
