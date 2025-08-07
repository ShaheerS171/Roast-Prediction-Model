import streamlit as st
import joblib
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt')
nltk.download('stopwords')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models using absolute paths
model = joblib.load(os.path.join(BASE_DIR, 'svc_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'tf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))



def preprocess_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = ''.join([ch for ch in txt if not ch.isdigit()])
    txt = ' '.join([word for word in txt.split() if not word.startswith('http') and not word.startswith('www.')])
    txt = ''.join([ch for ch in txt if ch.isascii()])
    words = word_tokenize(txt)
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in words if word not in stop_words]
    return ' '.join(cleaned)


st.set_page_config(page_title="Roast Classifier", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #23272f;}
    .stTextArea textarea {
        font-size: 16px;
        background-color: #2d333b;
        color: #f4f4f4;
    }
    .prediction-box {
        background-color: #ffb74d;
        padding: 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        color: #23272f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔥 Roast Text Classifier")
st.markdown("Enter any sentence or roast and the model will predict its label based on trained NLP pipeline.")

with st.form("roast_form"):
    user_input = st.text_area("Enter your text here:", height=200, placeholder="Type your roast here...")
    submitted = st.form_submit_button("Classify")

if submitted:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_input)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="prediction-box">Prediction: <span style="color:#23272f;">{}</span></div>'.format(predicted_label), unsafe_allow_html=True)

        # Optional: Show probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized_input)[0]
            labels = label_encoder.classes_
            st.subheader("Prediction Probabilities:")
            st.bar_chart(dict(zip(labels, proba)))

# Footer
st.markdown("---")
st.markdown("📍 Model: Support Vector Classifier | TF-IDF | Stopword Removal | Custom Preprocessing", unsafe_allow_html=True)