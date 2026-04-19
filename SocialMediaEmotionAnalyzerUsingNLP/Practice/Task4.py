from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "machine learning amazing",
    "love natural language processing",
    "machine learning love"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X.toarray())
