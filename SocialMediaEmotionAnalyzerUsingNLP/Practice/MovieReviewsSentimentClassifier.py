import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import text


url = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv"
df = pd.read_csv(url).head(5000)
print(f"Number of rows in the dataset: {len(df)}")
print(f"Columns in the dataset: {df.columns.tolist()}")
print(df.head(10))

base_stop_words = list(text.ENGLISH_STOP_WORDS)
words_to_keep = ['not', 'no', 'never', 'neither', 'nor']
my_stop_words = [word for word in base_stop_words if word not in words_to_keep]
my_stop_words.append("another")

tfidf = TfidfVectorizer(stop_words = my_stop_words, max_features=2500,ngram_range=(1, 2))
X = tfidf.fit_transform(df['reviewText'])
y = df['Positive']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

print("\n" + "="*40)
print(" REAL-TIME SENTIMENT ANALYZER (SMOTE Model)")
print("="*40)
print("Instructions: Enter your review below. Type 'exit' to stop.")

while True:

    user_review = input("\nWrite your review: ")
    if user_review.lower() == 'exit':
        print("Exiting... Good job on completing the practice!")
        break

    review_vector = tfidf.transform([user_review])
    prediction = model.predict(review_vector)[0]
    

    if prediction == 1:
        print("Result: Positive 😊")
    else:
        print("Result: Negative 😠")

    
    prob = model.predict_proba(review_vector)[0]
    confidence = max(prob) * 100
    print(f"Confidence: {confidence:.2f}%")


