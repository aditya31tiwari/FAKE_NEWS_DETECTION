# app.py
# Ensemble Fake News Detection:
# SVM (TF-IDF) + Naive Bayes (TF-IDF) + BERT + XGBoost
# 0 = FAKE, 1 = REAL everywhere.

import os
import re
from collections import Counter

import joblib
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DetectoNews", page_icon="🕵️", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TFIDF_FILE = "tfidf_vectorizer_new.pkl"
SVM_FILE = "svm_model_v1.pkl"
NB_FILE = "naive_bayes_model_v2.pkl"
BERT_WRAPPER_FILE = "bert_xgb_wrapper.pkl"
LABEL_ENCODER_FILE = "label_encoder.joblib"  

LABEL_FAKE = 0
LABEL_REAL = 1


# ---------- Helpers ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {filename}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {filename}: {e}")
        return None


def idx_to_label(idx: int) -> str:
    return "FAKE" if idx == LABEL_FAKE else "REAL"


def safe_predict_classic(model, X):
    """
    For SVM / NB on TF-IDF.
    Returns (pred_idx, confidence) where pred_idx in {0,1}.
    Confidence is probability of the predicted class.
    """
    if model is None:
        return None, None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]  # [P(fake), P(real)]
            pred_idx = int(np.argmax(proba))
            conf = float(proba[pred_idx])
            return pred_idx, conf
        # fallback: use decision_function or predict only
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # sigmoid-ish mapping for binary
            if np.ndim(scores) == 1:
                s = float(scores[0])
                prob_real = 1.0 / (1.0 + np.exp(-s))
                prob_fake = 1.0 - prob_real
                proba = np.array([prob_fake, prob_real])
                pred_idx = int(np.argmax(proba))
                return pred_idx, float(proba[pred_idx])
        pred_idx = int(model.predict(X)[0])
        return pred_idx, 1.0
    except Exception:
        return None, None


# ---------- Cached loaders ----------
@st.cache_resource
def load_models():
    tfidf = safe_load(TFIDF_FILE)
    svm = safe_load(SVM_FILE)
    nb = safe_load(NB_FILE)
    bert_wrapper = safe_load(BERT_WRAPPER_FILE)
    le = safe_load(LABEL_ENCODER_FILE)
    return tfidf, svm, nb, bert_wrapper, le


@st.cache_resource
def load_bert_model(model_name: str):
    return SentenceTransformer(model_name)


tfidf, svm_model, nb_model, bert_wrapper, label_enc = load_models()


def predict_bert_xgb(text: str):
    """
    Uses wrapper: {"clf": xgb_model, "label_encoder": ..., "bert_name": "..."}
    Returns (pred_idx, confidence).
    """
    if bert_wrapper is None or not isinstance(bert_wrapper, dict):
        return None, None

    clf = bert_wrapper.get("clf")
    bert_name = bert_wrapper.get("bert_name", "all-mpnet-base-v2")
    if clf is None:
        return None, None

    try:
        bert_model = load_bert_model(bert_name)
        emb = bert_model.encode([text.strip()], convert_to_numpy=True)
        pred_idx = int(clf.predict(emb)[0])

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(emb)[0]  # [P(fake), P(real)]
            conf = float(proba[pred_idx])
        else:
            conf = 1.0

        return pred_idx, conf
    except Exception as e:
        st.error(f"BERT+XGBoost prediction error: {e}")
        return None, None


# ---------- UI ----------
st.title("DetectoNews🕵️")
st.write("This app uses three models SVM, Naive Bayes and an Hybrid Model, i.e. BERT+XGBoost. "
         "0 = FAKE, 1 = REAL. Final verdict is by majority vote.")

text_input = st.text_area("Enter news article text:", height=200, placeholder="Paste the news article here...")

if st.button("Check"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        raw_text = text_input.strip()
        cleaned = clean_text(raw_text)

        if tfidf is None:
            st.error("TF-IDF vectorizer not loaded.")
        else:
            X_tfidf = tfidf.transform([cleaned])

            # Individual model predictions
            svm_idx, svm_conf = safe_predict_classic(svm_model, X_tfidf)
            nb_idx, nb_conf = safe_predict_classic(nb_model, X_tfidf)
            bert_idx, bert_conf = predict_bert_xgb(raw_text)

            svm_label = idx_to_label(svm_idx) if svm_idx in (0, 1) else "ERROR"
            nb_label = idx_to_label(nb_idx) if nb_idx in (0, 1) else "ERROR"
            bert_label = idx_to_label(bert_idx) if bert_idx in (0, 1) else "ERROR"

            # Ensemble voting
            votes = [i for i in (svm_idx, nb_idx, bert_idx) if i in (0, 1)]
            if votes:
                counts = Counter(votes)  # e.g. {0:2, 1:1}
                final_idx, _ = counts.most_common(1)[0]
                final_label = idx_to_label(final_idx)

                # Main result
                st.markdown("---")
                st.subheader(f"Final Verdict: **{final_label}**")
                if final_label == "FAKE":
                    st.error("This article is likely FAKE!!.")
                else:
                    st.success("This article is likely REAL!!.")

                st.write(f"Votes (0 = FAKE, 1 = REAL): {dict(counts)}")

                # Details expander
                with st.expander("Ensemble model details"):
                    st.markdown("### Individual Model Predictions")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**SVM Prediction**")
                        st.metric(label="", value=f"{svm_label}")
                        if svm_conf is not None:
                            st.success(f"Confidence: {svm_conf:.2f}")

                    with col2:
                        st.markdown("**Naive Bayes Prediction**")
                        st.metric(label="", value=f"{nb_label}")
                        if nb_conf is not None:
                            st.success(f"Confidence: {nb_conf:.2f}")

                    with col3:
                        st.markdown("**BERT+XGBoost Prediction**")
                        st.metric(label="", value=f"{bert_label}")
                        if bert_conf is not None:
                            st.success(f"Confidence: {bert_conf:.2f}")

                    if isinstance(bert_wrapper, dict):
                        st.caption(f"Transformer used: `{bert_wrapper.get('bert_name', 'unknown')}`")

            else:
                st.error("No valid predictions available for ensemble. Check that all models loaded correctly.")


