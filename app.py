# app.py
"""
Streamlit app for BERT + XGBoost hybrid.
Default transformer: 'all-MiniLM-L6-v2' (as you specified).
Expects 'bert_xgb_wrapper.pkl' (dict with keys like 'model'/'clf' and optional 'bert_name'/'transformer_name'/'label_encoder').
"""

import os
import joblib
import numpy as np
import streamlit as st
from typing import Any

# ---------- Configuration ----------
WRAPPER_FILENAME = "bert_xgb_wrapper.pkl"
DEFAULT_BERT_NAME = "all-MiniLM-L6-v2"  # <- your transformer name

st.set_page_config(page_title="DetectoNews — BERT+XGBoost", page_icon="🧠", layout="centered")


# ---------- Helpers ----------
@st.cache_resource
def load_wrapper(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        return {"__load_error__": str(e)}

@st.cache_resource
def get_transformer(model_name: str):
    # cached SentenceTransformer loader
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def extract_from_wrapper(obj: Any):
    clf = None
    le = None
    bert_name = None

    if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
        return obj, getattr(obj, "le", None), getattr(obj, "bert_name", DEFAULT_BERT_NAME)

    if isinstance(obj, dict):
        for k in ("model", "clf", "xgb", "classifier"):
            if k in obj:
                clf = obj[k]; break
        for k in ("label_encoder", "le"):
            if k in obj:
                le = obj[k]; break
        for k in ("bert_name", "transformer_name", "hf_model_name"):
            if k in obj:
                bert_name = obj[k]; break
        if clf is None:
            for v in obj.values():
                if hasattr(v, "predict") and hasattr(v, "predict_proba"):
                    clf = v; break

    if clf is None:
        return None, None, None
    if bert_name is None:
        bert_name = DEFAULT_BERT_NAME
    return clf, le, bert_name

def safe_proba_from_clf(clf, embeddings):
    try:
        p = clf.predict_proba(embeddings)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return float(np.clip(p[0, 1], 0.0, 1.0))
        return float(np.clip(p.ravel()[0], 0.0, 1.0))
    except Exception:
        try:
            df = clf.decision_function(embeddings)
            df = np.asarray(df).ravel()
            score = df[0]
            return float(1.0 / (1.0 + np.exp(-score)))
        except Exception:
            return None

def decode_label(raw_pred, le):
    if le is not None:
        try:
            return str(le.inverse_transform([raw_pred])[0])
        except Exception:
            try:
                return str(le[raw_pred])
            except Exception:
                pass
    try:
        return "REAL" if int(raw_pred) == 1 else "FAKE"
    except Exception:
        return str(raw_pred)


# ---------- Load wrapper ----------
st.title("DetectoNews — BERT + XGBoost (hybrid)")
cwd = os.getcwd()
st.sidebar.header("Runtime info")
st.sidebar.write("Working dir:", cwd)
try:
    st.sidebar.write("Files:", os.listdir(cwd)[:200])
except Exception:
    pass

loaded = load_wrapper(os.path.join(cwd, WRAPPER_FILENAME))
if loaded is None:
    st.error(f"Wrapper file '{WRAPPER_FILENAME}' not found in working directory.")
    st.stop()
if isinstance(loaded, dict) and "__load_error__" in loaded:
    st.error(f"Failed to load wrapper file: {loaded['__load_error__']}")
    st.stop()

clf, label_enc, bert_name = extract_from_wrapper(loaded)
if clf is None:
    st.error("Could not find a classifier inside the wrapper. Ensure wrapper dict contains 'model'/'clf' or similar.")
    st.stop()

st.sidebar.write("Classifier type:", type(clf))
st.sidebar.write("Label encoder present:", label_enc is not None)
st.sidebar.write("Transformer to be used:", bert_name)

# ---------- UI ----------
st.write("Paste a news article/claim below and press **Check**. (First run loads transformer and may take a few seconds.)")
text = st.text_area("Article / Claim", height=240, placeholder="Paste the news text here...")

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Embedding text with SentenceTransformer and predicting..."):
            try:
                transformer = get_transformer(bert_name)
            except Exception as e:
                st.error(f"Failed to load transformer '{bert_name}': {e}")
                st.stop()

            try:
                embeddings = transformer.encode([text.strip()], show_progress_bar=False)
                embeddings = np.asarray(embeddings)
            except Exception as e:
                st.error(f"Failed to create embeddings: {e}")
                st.stop()

            try:
                raw_pred = clf.predict(embeddings)[0]
            except Exception as e:
                st.error(f"Classifier predict failed: {e}")
                st.stop()

            conf = safe_proba_from_clf(clf, embeddings)
            if conf is None:
                conf = 0.5

            label = decode_label(raw_pred, label_enc)

            st.markdown("---")
            st.subheader(f"Prediction: **{label}**")
            st.metric("Confidence", f"{int(round(conf * 100))} %")
            st.progress(conf)
            if str(label).lower().strip() in {"fake", "0", "false"}:
                st.warning("This content is flagged as likely FAKE. Verify with trusted sources.")
            else:
                st.success("This content is flagged as likely REAL — still verify high-stakes claims.")

with st.expander("Debug info"):
    st.write("Loaded wrapper type:", type(loaded))
    if isinstance(loaded, dict):
        st.write("Wrapper keys:", list(loaded.keys()))
    st.write("Classifier type:", type(clf))
    st.write("Transformer name:", bert_name)
    st.write("Working dir:", cwd)
