import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import time


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { border-radius: 12px; border: 1px solid #ced4da; font-size: 16px; }
    .prediction-card { padding: 25px; border-radius: 15px; text-align: center; margin-top: 25px; font-family: 'Segoe UI', sans-serif; }
    .fake-result { background-color: #fff5f5; color: #d32f2f; border: 2px solid #ffcdd2; }
    .real-result { background-color: #f1f8e9; color: #2e7d32; border: 2px solid #c8e6c9; }
    .status-text { font-weight: bold; font-size: 24px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():

    model = tf.keras.models.load_model('Models/fake_news_lstm_model.h5')
    
    with open('Models/tokenizer.pickle', 'rb') as handle:
        loaded_data = pickle.load(handle)

    if isinstance(loaded_data, dict):
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.word_index = loaded_data
    else:
        tokenizer = loaded_data
        
    return model, tokenizer

model, tokenizer = load_assets()
def clean_and_prepare(text, tokenizer, max_length=300):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    clean_text = " ".join(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    return padded
st.title("Fake News Detector")
st.markdown("#### *Deep Learning Powered Fake News Detection*")
st.write("This application uses a **Bidirectional LSTM** network to analyze the context of news articles.")

st.divider()

user_input = st.text_area("News Article Content", height=250, placeholder="Paste the news text here to analyze...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("Analyze News", use_container_width=True)

if predict_btn:
    if not user_input.strip():
        st.warning(" Please enter the news content to proceed.")
    else:
        with st.spinner('Neural Network is scanning for misinformation patterns...'):
            processed_input = clean_and_prepare(user_input, tokenizer)
            prediction_prob = model.predict(processed_input)[0][0]
            time.sleep(0.8) 
        
            if prediction_prob > 0.8:
                confidence = prediction_prob * 100
                st.markdown(f"""
                    <div class="prediction-card fake-result">
                        <div class="status-text">Prediction: FAKE NEWS</div>
                        <p>The model is <b>{confidence:.2f}%</b> confident that this content contains misinformation.</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(float(prediction_prob))
            else:
                confidence = (1 - prediction_prob) * 100
                st.markdown(f"""
                    <div class="prediction-card real-result">
                        <div class="status-text">Prediction: REAL NEWS</div>
                        <p>The model is <b>{confidence:.2f}%</b> confident that this content is authentic.</p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(float(1 - prediction_prob))

st.divider()
st.info("**Tip:** Long-form articles provide more context for the LSTM layers, leading to higher accuracy.")

with st.expander(" Technical Details"):
    st.write("""
    - **Architecture:** Bidirectional LSTM with Dropout layers.
    - **Vocabulary Size:** 10,000 most frequent tokens.
    - **Input Length:** 300 words.
    - **Optimization:** Adam Optimizer.
    """)