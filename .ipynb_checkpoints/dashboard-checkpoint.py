import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

nltk.download('stopwords')

# ------------------- Text Cleaning -------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# ------------------- Load Dataset -------------------
df = pd.read_csv("AI Tools Dataset.csv")

# Keep only required columns
df = df[['Short Description','Category']].dropna()

df['clean_text'] = df['Short Description'].apply(clean_text)

# ------------------- Train Model -------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['Category']

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="AI Tools NLP Dashboard", layout="centered")

st.title("🧠 AI Tool Category Prediction System")
st.subheader("Predict AI Tool Category using NLP + ML")

user_input = st.text_area("✍ Enter AI Tool Description:")

if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some description!")
    else:
        clean = clean_text(user_input)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]

        st.success(f"🔮 Predicted Category: **{prediction}**")

