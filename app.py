import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="centered")

# --- Professional CSS styling ---
st.markdown("""
    <style>
    /* Body background */
    body {
        background-color: #f4f6f9;
    }

    .stApp {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title styling */
    .title {
        font-size: 2.3rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Textarea styling */
    textarea {
        border-radius: 8px !important;
        padding: 10px !important;
        border: 1px solid #d0d4db !important;
    }

    /* Predict button */
    div.stButton > button:first-child {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 1em;
        font-weight: 600;
        margin-top: 10px;
    }

    div.stButton > button:hover {
        background-color: #3e4f61;
        color: #fff;
    }

    /* Prediction box */
    .result-box {
        background-color: #ecf0f1;
        padding: 1rem;
        border-left: 5px solid #3498db;
        border-radius: 8px;
        margin-top: 1.5rem;
        font-size: 1.1rem;
    }

    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<div class='title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("This ML-powered tool predicts whether the news you paste below is **REAL** or **FAKE**.")

# --- Input ---
news = st.text_area("Paste your news article here", height=200)

# --- Predict Button ---
if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        prob = model.predict_proba(vec)

        label = pred[0]
        confidence = round(100 * max(prob[0]), 2)

        result_color = "#27ae60" if label == "REAL" else "#e74c3c"
        result_text = f"✅ REAL News (Confidence: {confidence}%)" if label == "REAL" else f"❌ FAKE News (Confidence: {confidence}%)"

        st.markdown(
            f"<div class='result-box' style='border-left: 5px solid {result_color}; color: {result_color};'>{result_text}</div>",
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown("<div class='footer'>Built by Reethika | Powered by Machine Learning</div>", unsafe_allow_html=True)

