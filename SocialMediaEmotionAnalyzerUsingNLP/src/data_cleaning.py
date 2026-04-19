import pandas as pd

df = pd.read_csv('../Data/twitter_training.csv', names=['id', 'entity', 'sentiment', 'text'])
df = df[df['sentiment'] != 'Irrelevant']
df['sentiment_label'] = df['sentiment'].map({'Positive':2 , 'Neutral':1, 'Negative':0})
df.dropna(subset='text', inplace= True)
print(df['sentiment'].value_counts())
df.to_csv('../Data/cleaned_twitter_training.csv', index=False)
print("Cleaning Done!")