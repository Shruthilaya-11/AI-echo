import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
from typing import Tuple, Dict

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="AI Echo — Sentiment Explorer", layout="wide")
st.title("🧠 AI Echo — Sentiment & EDA Explorer")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Shruthilaya\GUVI\data\chatgpt_style_reviews_dataset.xlsx - Sheet1.csv")
    df = df[['date', 'platform', 'review', 'rating', 'helpful_votes', 'verified_purchase']].dropna()
    return df

df = load_data()
st.markdown("### Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Sentiment Labeling
# ---------------------------
def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['rating'].apply(label_sentiment)

# ---------------------------
# Exploratory Data Analysis
# ---------------------------
st.header("Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Rating Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='rating', data=df, ax=ax, palette="coolwarm")
    st.pyplot(fig)

with col2:
    st.subheader("Sentiment Breakdown")
    fig, ax = plt.subplots()
    sns.countplot(x='sentiment', data=df, ax=ax, palette="viridis")
    st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Verified vs Non-Verified Average Ratings")
    fig, ax = plt.subplots()
    sns.barplot(x='verified_purchase', y='rating', data=df, ax=ax)
    st.pyplot(fig)

with col4:
    st.subheader("Platform-wise Average Ratings")
    fig, ax = plt.subplots()
    sns.barplot(x='platform', y='rating', data=df, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# ---------------------------
# NLTK Setup for Text Preprocessing
# ---------------------------
_required_nltk = ["punkt", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]
for pkg in _required_nltk:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text: str, keep_case: bool=False) -> str:
    if text is None:
        return ""
    text = str(text)
    if not keep_case:
        text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOPWORDS]
    return " ".join(filtered)

df['clean_review'] = df['review'].astype(str).apply(clean_text)

# ---------------------------
# Load VADER & Transformers
# ---------------------------
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_transformer_pipeline(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model_name, framework="pt", device=device)

vader = load_vader()
try:
    bert_pipe = load_transformer_pipeline()
    hf_load_error = None
except Exception as e:
    bert_pipe = None
    hf_load_error = e

# ---------------------------
# Utility Functions
# ---------------------------
def vader_result_to_label(score_dict: Dict[str, float]) -> Tuple[str, float]:
    comp = score_dict.get("compound", 0.0)
    if comp >= 0.05:
        return "Positive", comp
    elif comp <= -0.05:
        return "Negative", comp
    else:
        return "Neutral", comp

def hf_label_normalize(hf_output):
    if isinstance(hf_output, list) and len(hf_output) > 0:
        out = hf_output[0]
        lbl = out.get("label", "")
        score = float(out.get("score", 0.0))
        if lbl.lower().startswith("pos"):
            return "Positive", score
        elif lbl.lower().startswith("neg"):
            return "Negative", score
        else:
            return lbl.title(), score
    return "Unknown", 0.0

# ---------------------------
# Streamlit UI — Sentiment Analysis
# ---------------------------
st.header("Live Sentiment Analysis — VADER vs DistilBERT")

user_text = st.text_area("Enter a review or sentence here:", height=160)

col1, col2 = st.columns(2)

with col1:
    if st.button("Run VADER"):
        if not user_text.strip():
            st.warning("Please enter text!")
        else:
            cleaned = clean_text(user_text)
            label_v, score_v = vader_result_to_label(vader.polarity_scores(user_text))
            st.metric("VADER label", label_v)
            st.write(f"Compound score: {score_v:.3f}")
            st.write("Full scores:", vader.polarity_scores(user_text))
            st.write("Cleaned text:", cleaned)

with col2:
    if st.button("Run DistilBERT"):
        if not user_text.strip():
            st.warning("Please enter text!")
        else:
            if hf_load_error:
                st.error(f"HF load error: {hf_load_error}")
            else:
                cleaned = clean_text(user_text)
                hf_out = bert_pipe(user_text)
                label_hf, score_hf = hf_label_normalize(hf_out)
                st.metric("DistilBERT label", label_hf)
                st.write(f"Confidence: {score_hf:.3f}")
                st.write("Model output:", hf_out)
                st.write("Cleaned text:", cleaned)

st.markdown("---")
st.subheader("Side-by-side Comparison")

if st.button("Run both"):
    if not user_text.strip():
        st.warning("Please enter text!")
    else:
        cleaned = clean_text(user_text)
        v_scores = vader.polarity_scores(user_text)
        v_label, v_comp = vader_result_to_label(v_scores)

        if hf_load_error:
            hf_label, hf_score = "Error", 0.0
        else:
            hf_out = bert_pipe(user_text)
            hf_label, hf_score = hf_label_normalize(hf_out)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### VADER")
            st.write(f"Label: {v_label}")
            st.write(f"Compound score: {v_comp:.3f}")
            st.write("Detailed scores:", v_scores)
        with c2:
            st.markdown("### DistilBERT")
            st.write(f"Label: {hf_label}")
            st.write(f"Confidence: {hf_score:.3f}")
            if not hf_load_error:
                st.write("Model output:", hf_out)

st.markdown("---")
st.caption("App combines EDA, text cleaning, and sentiment comparison (VADER vs DistilBERT). LSTM modeling/training can be plugged in here if needed.")
