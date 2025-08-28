import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from wordcloud import WordCloud
import numpy as np

st.set_page_config(page_title="AI Echo — Sentiment Explorer",layout="wide")
st.title("AI Echo — Sentiment Explorer")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Shruthilaya\GUVI\data\chatgpt_style_reviews_dataset.xlsx - Sheet1.csv")
    # Fix date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

df = load_data()

# Sentiment Labeling
#1-2 = negative, 3 = neutral, 4-5 = positive
def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["rating"].apply(label_sentiment)


# EDA 
st.subheader("EDA Insights")

# Helpful Reviews + Word Clouds
col1, col2 = st.columns(2)

with col1:
    st.markdown("###Helpful Reviews")
    helpful_count = (df["helpful_votes"] > 10).sum()
    not_helpful_count = (df["helpful_votes"] <= 10).sum()
    fig, ax = plt.subplots()
    ax.pie([helpful_count, not_helpful_count],
           labels=["Helpful", "Not Helpful"],
           autopct="%1.1f%%",
           colors=["green", "red"])
    st.pyplot(fig)

with col2:
    st.markdown("###Common Keywords (Positive vs Negative)")
    positive_reviews = " ".join(df[df["rating"] >= 4]["review"].astype(str))
    negative_reviews = " ".join(df[df["rating"] <= 2]["review"].astype(str))

    pos_wc = WordCloud(width=600, height=400, background_color="white").generate(positive_reviews)
    neg_wc = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(negative_reviews)

    subcol1, subcol2 = st.columns(2)
    with subcol1:
        st.markdown("**Positive**")
        st.image(pos_wc.to_array(), use_container_width=True)
    with subcol2:
        st.markdown("**Negative**")
        st.image(neg_wc.to_array(), use_container_width=True)

#Ratings Over Time + Rating Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("###Average Rating Over Time")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    trend = df.groupby("date")["rating"].mean()
    fig, ax = plt.subplots()
    trend.plot(ax=ax)
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

with col2:
    st.markdown("###Rating Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="rating", data=df, ax=ax)
    st.pyplot(fig)  

# Platform + Verified + Sentiment breakdown
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Ratings by Platform")
    fig, ax = plt.subplots()
    sns.barplot(x="platform", y="rating", data=df, ax=ax, estimator=np.mean)
    st.pyplot(fig)

with col2:
    st.markdown("###Verified vs Non-Verified")
    fig, ax = plt.subplots()
    sns.barplot(x="verified_purchase", y="rating", data=df, ax=ax, estimator=np.mean)
    st.pyplot(fig)

with col3:
    st.write("### Sentiment Breakdown")
    fig, ax = plt.subplots()
    sns.countplot(x="sentiment", data=df, ax=ax)
    st.pyplot(fig)

# Review Length + 1-Star Words
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Review Length per Rating")
    df["review_length"] = df["review"].astype(str).apply(len)
    fig, ax = plt.subplots()
    sns.boxplot(x="rating", y="review_length", data=df, ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("### Common Words in 1-Star Reviews")
    one_star_text = " ".join(df[df["rating"] == 1]["review"].astype(str))
    one_wc = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(one_star_text)
    st.image(one_wc.to_array(), use_container_width=True)


# ML Section
st.subheader("ML Model — Sentiment Classification")

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model_choice = st.radio("Pick a model:", ["Naive Bayes", "Logistic Regression", "Linear SVM"])

if model_choice == "Naive Bayes":
    model = MultinomialNB()
elif model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = LinearSVC()

model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.text(classification_report(y_test, y_pred))

# AUC
try:
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(pd.get_dummies(y_test), model.predict_proba(X_test_vec), multi_class="ovr")
    elif hasattr(model, "decision_function"):
        auc = roc_auc_score(pd.get_dummies(y_test), model.decision_function(X_test_vec), multi_class="ovr")
    st.write("Macro AUC:", round(auc, 3))
except Exception as e:
    st.warning(f"AUC not available for this model: {e}")

user_text = st.text_area("Write a fake review:")
if st.button("Analyze") and user_text.strip() != "":
    user_vec = vectorizer.transform([user_text])
    pred = model.predict(user_vec)[0]
    st.success(f"Predicted sentiment: **{pred}**")
