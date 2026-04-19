import pandas as pd

cols = ['id', 'entity', 'sentiment', 'text']
df = pd.read_csv('twitter_training.csv',names = cols)
print(df.head(10))
print(df.describe())
print(df.info())
print(f"Missing values in each column:\n{df.isnull().sum()}")
print("="*30)
print("1. DATASET COLUMNS:")
print(df.columns.tolist())
print("="*30)
counts = df['sentiment'].value_counts()
print(counts)
print(df['sentiment'].value_counts(normalize=True) * 100)
print("="*30)