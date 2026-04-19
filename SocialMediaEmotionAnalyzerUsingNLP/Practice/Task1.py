import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#nltk.download('punkt')
#nltk.download('stopwords')

text = "Wow! Machine Learning is amazing, isn't it? I love NLP."

tokens= word_tokenize(text)
stop_words = set(stopwords.words('english'))
cleaned_tokens=[w for w in tokens if w.lower() not in stop_words and w.isalpha()]

print(f"before cleaning tokens: {tokens}")
print(f"After cleaning: {cleaned_tokens}")