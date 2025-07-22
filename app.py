import torch
import types

# Patch to avoid Streamlit crashing on torch.classes inspection
if not hasattr(torch, "__path__"):
    torch.__path__ = types.SimpleNamespace(_path=[])

import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer, util

# --- Setup ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words("english"))

# --- Label Maps ---
intent_labels = {
    0: 'Debugging',
    1: "How-to",
    2: "Concept"}
difficulty_label_map = {
    0: "Easy",
    1: "Medium",
    2: "Hard"}

# --- Clean Text Function ---
def clean_text(text, max_words=None):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = text.replace("`", "")
    text = re.sub(r"^[=\-*#~_]{3,}", "", text, flags=re.MULTILINE)
    text = re.sub(r"[{}\[\]\"']", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\-+#_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Load Models & Data ---
base_path = os.path.dirname(os.path.abspath(__file__))

tag_model = joblib.load(os.path.join(base_path, "tag pred model.pkl"))
mlb = joblib.load(os.path.join(base_path, "mlb.pkl"))
difficulty_model = joblib.load(os.path.join(base_path, "difficulty pred model.pkl"))
intent_model = joblib.load(os.path.join(base_path, "intent pred model.pkl"))
corpus_embeddings = np.load(os.path.join(base_path, "corpus_embeddings.npy"))

# Load corpus texts for similarity display
df = pd.read_csv(os.path.join(base_path, "questions.csv")) 
question_texts = df["clean_text"].tolist()

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="AutoTagger", layout="wide")
st.title("AutoTagger: Smart Question Analyzer")

user_input = st.text_area("Enter your question:", height=180)

if st.button("Analyze"):
    if user_input.strip():
        clean_input = clean_text(user_input)

        # --- Tag Prediction ---
        tag_probs = tag_model.predict_proba([clean_input])[0]
        tag_classes = mlb.classes_
        tag_confidence = sorted(zip(tag_classes, tag_probs), key=lambda x: x[1], reverse=True)
        top_tags = [(tag, prob) for tag, prob in tag_confidence if prob > 0.5]

        # --- Difficulty Prediction ---
        difficulty = difficulty_model.predict([clean_input])[0]
        diff_prob = None
        if hasattr(difficulty_model, "predict_proba"):
            diff_prob = np.max(difficulty_model.predict_proba([clean_input]))

        # --- Intent Prediction ---
        intent = intent_model.predict([clean_input])[0]
        intent_prob = None
        if hasattr(intent_model, "predict_proba"):
            intent_prob = np.max(intent_model.predict_proba([clean_input]))

        # --- Similar Questions ---
        input_embedding = embedder.encode([clean_input], convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, corpus_embeddings)[0]
        top_k = min(3, len(scores))
        top_indices = scores.topk(k=top_k).indices.cpu().numpy()

        # --- Layout Sections ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("Predicted Tags")
            if top_tags:
                for tag, prob in top_tags:
                    st.markdown(f"**{tag}** — `{prob:.2f}`")
                    progress_bar = st.progress(0)
                    for percent in range(0, int(prob * 100) + 1, 5):
                        progress_bar.progress(percent / 100.0)
                        time.sleep(0.01)
            else:
                st.write("No high-confidence tags found.")

        with col2:
            st.markdown("Predicted Difficulty")
            label_num = difficulty_label_map.get(difficulty, -1)
            st.markdown(f"**{label_num}**")
            if diff_prob:
                progress_bar = st.progress(0)
                for percent in range(0, int(diff_prob * 100) + 1, 5):
                    progress_bar.progress(percent / 100.0)
                    time.sleep(0.01)

        with col3:
            st.markdown("Predicted Intent")
            label_num = intent_labels.get(intent, -1)
            st.markdown(f"**{label_num}**")
            if intent_prob:
                progress_bar = st.progress(0)
                for percent in range(0, int(intent_prob * 100) + 1, 5):
                    progress_bar.progress(percent / 100.0)
                    time.sleep(0.01)

        # --- Similar Questions ---
        st.markdown("---")
        st.subheader("Top Similar Questions")
        for idx in top_indices:
            score = scores[idx].item()
            question = question_texts[idx]
            st.markdown(
                f"**Similarity:** `{score:.2f}` — {question}")
