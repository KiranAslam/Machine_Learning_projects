import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocessing


df = pd.read_csv('../Data/cleaned_twitter_training.csv')
df.drop_duplicates(subset=['text'], inplace=True)
df['cleaned_text'] = df['text'].apply(preprocessing)

X_train , X_test , y_train , y_test = train_test_split(
    df['cleaned_text'], df['sentiment_label'], test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(ngram_range =(1,3), max_features=100000, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model ={
    'Logistic Regression': LogisticRegression(max_iter=1000, C = 2.0),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    'Linear SVC' : LinearSVC(C=0.1, max_iter=2000, loss='hinge')
}

best_accuracy=0
best_model = None

for name, model in model.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Name :{name}")
    print(f"Accuracy :{accuracy *100 :.2f}")
    print(f"Classification Report :\n{classification_report(y_test, y_pred)}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"Best Model : {best_model_name} with Accuracy : {best_accuracy * 100:.2f}%")
if best_accuracy > 0.80:
        joblib.dump(best_model, '../Models/best_emotion_model.pkl')
        joblib.dump(vectorizer, '../Models/tfidf_vectorizer.pkl')
        print("Model and Vectorizer saved successfully!")