import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

#nltk.download('stopwords')
#nltk.download('wordnet')

def load_data():
    df = pd.read_csv('../Data/WELFake_Dataset.csv')

    #print(df.head(10))
    #print(df.info())
    #print(df.describe())
    #print(df.isnull().sum())
    #print(df.duplicated().sum())
    #print(df['label'].value_counts())
    return df

def preprocess_data(df):

    df.fillna(' ', inplace=True)
    df['content'] = df['title'] + ' ' + df['text']
    df = df.drop_duplicates()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):

        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.lower().split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['content'] = df['content'].apply(preprocess_text)
    return df[['content','label']]

df = load_data()
df = preprocess_data(df)

def prepare_sequences(df):
    max_vocab = 10000
    max_len = 300
    tokenizer = Tokenizer(num_words= max_vocab, oov_token='<OOV>')

    tokenizer.fit_on_texts(df['content'])
    sequences  = tokenizer.texts_to_sequences(df['content'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences, np.array(df['label']), tokenizer

X, y, tokenizer_obj = prepare_sequences(df)
np.save('../Data/X_data.npy', X)
np.save('../Data/y_data.npy', y)
with open('../Models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Preprocessing completed and tokenizer saved successfully.")



