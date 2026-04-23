import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import re
import sys
import os

@st.cache_resource
def load_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

load_nltk_data()
sys.path.append(os.path.abspath("src"))
from src.preprocessing import preprocessing
model = joblib.load('Models/best_emotion_model.pkl')
vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')

st.set_page_config(page_title = "EmotionAnalyzer")
st.title("Social Media Emotion Analyzer")
st.markdown("Enter a tweet or comment below to analyze its sentiment**.")

user_input = st.text_area("What's on your mind?", placeholder="e.g., I'm not happy with the new update! :(")
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
      
        cleaned_text = preprocessing(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        scores = model.decision_function(vectorized_text)[0]
        probs = softmax(scores)
         
        st.subheader("Analysis Result:")
        if prediction == 0:
            st.error(" Sentiment: **NEGATIVE**")
        elif prediction == 1:
            st.info(" Sentiment: **NEUTRAL**")
        else:
            st.success(" Sentiment: **POSITIVE**")

        labels = ['Negative', 'Neutral', 'Positive']
        fig = px.pie(values=probs, names=labels, 
                     title="Emotion Probability Distribution",
                     color=labels,
                     color_discrete_map={'Negative':'#EF553B', 'Neutral':'#636EFA', 'Positive':'#00CC96'},
                     hole=0.4)
        st.plotly_chart(fig)

        with st.expander("See Technical Details"):
            st.write(f"**Cleaned Text:** `{cleaned_text}`")
            st.write(f"**Confidence Scores:** {dict(zip(labels, np.round(probs*100, 2)))} %")
    

st.divider()
st.caption("Powered by NLP & Linear SVM | Built with ❤️ by Kiran Aslam")
