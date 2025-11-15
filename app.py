# app.py
"""
Streamlit app for BERT + XGBoost hybrid.
Expects bert_xgb_wrapper.pkl in the working directory. The wrapper should be a dict
with keys: "clf", "label_encoder", and "bert_name" (e.g., "all-mpnet-base-v2").
"""

import os
import traceback
import joblib
import numpy as np
import streamlit as st

# ---------- Config ----------
WRAPPER_FILENAME = "bert_xgb_wrapper.pkl"
DEFAULT_BERT_NAME = "all-mpnet-base-v2"  # fallback if wrapper lacks bert_name

# ---------- Streamlit cache compatibility ----------
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache

# ---------- Helpers ----------
@cache_resource
def load_wrapper(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return {"__load_error__": traceback.format_exc()}

@cache_resource
def get_transformer(model_name: str):
    # Load SentenceTransformer (accepts HF name or local path)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def extract_from_wrapper(obj):
    """Return (clf, label_encoder, bert_name, embedding_dim) or (None,...)."""
    clf = None
    le = None
    bert_name = None
    emb_dim = None

    if isinstance(obj, dict):
        clf = obj.get("clf") or obj.get("model") or obj.get("xgb") or obj.get("classifier")
        le = obj.get("label_encoder") or obj.get("le")
        bert_name = obj.get("bert_name") or obj.get("transformer_name") or obj.get("hf_model_name")
        emb_dim = obj.get("embedding_dim") or (obj.get("transformer_info") or {}).get("embedding_dim")
        # fallback: if not found, try to discover any value that looks like a clf
        if clf is None:
            for v in obj.values():
                if hasattr(v, "predict"):
                    clf = v
                    break
    else:
        # If the wrapper itself is a model object
        if hasattr(obj, "predict"):
            clf = obj

    if bert_name is None:
        bert_name = DEFAULT_BERT_NAME
    return clf, le, bert_name, emb_dim

from scipy.special import expit
def safe_proba_from_clf(clf, X):
    """
    Try predict_proba -> decision_function -> xgboost.Booster fallback -> predict.
    Returns float in [0,1] or None.
    """
    try:
        Xa = np.asarray(X)
        if Xa.ndim == 1:
            Xa = Xa.reshape(1, -1)
    except Exception:
        return None

    try:
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(Xa)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return float(np.clip(p[0, 1], 0.0, 1.0))
            return float(np.clip(p.ravel()[0], 0.0, 1.0))

        if hasattr(clf, "decision_function"):
            df = clf.decision_function(Xa)
            df = np.asarray(df).ravel()
            return float(expit(df[0]))

        # try xgboost Booster fallback
        try:
            import xgboost as xgb
            if hasattr(clf, "get_booster") or isinstance(clf, xgb.Booster):
                booster = clf.get_booster() if hasattr(clf, "get_booster") else clf
                dmat = xgb.DMatrix(Xa)
                raw = booster.predict(dmat)
                return float(expit(np.asarray(raw).ravel()[0]))
        except Exception:
            pass

        # last resort: try predict and coerce
        if hasattr(clf, "predict"):
            pred = clf.predict(Xa)
            pred = np.asarray(pred).ravel()[0]
            try:
                return float(np.clip(float(pred), 0.0, 1.0))
            except Exception:
                return None

    except Exception:
        return None

    return None

def decode_label(raw_pred, le):
    try:
        if le is not None:
            if hasattr(le, "inverse_transform"):
                return str(le.inverse_transform([raw_pred])[0])
            if isinstance(le, dict):
                return str(le.get(raw_pred, raw_pred))
            if isinstance(le, (list, tuple)):
                return str(le[int(raw_pred)])
        return "REAL" if int(raw_pred) == 1 else "FAKE"
    except Exception:
        return str(raw_pred)

# ---------- App layout ----------
st.set_page_config(page_title="DetectoNews — BERT+XGBoost", page_icon="🧠")
st.title("DetectoNews — BERT + XGBoost")

st.sidebar.header("Runtime info")
cwd = os.getcwd()
st.sidebar.write("Working dir:", cwd)
try:
    st.sidebar.write("Files:", os.listdir(cwd)[:200])
except Exception:
    pass

# ---------- Load wrapper ----------
loaded = load_wrapper(os.path.join(cwd, WRAPPER_FILENAME))
if loaded is None:
    st.error(f"Wrapper file '{WRAPPER_FILENAME}' not found in working directory.")
    st.stop()

if isinstance(loaded, dict) and "__load_error__" in loaded:
    st.error("Failed to load wrapper (see Debug info for full traceback).")

clf, label_enc, bert_name_from_wrapper, embedding_dim = extract_from_wrapper(loaded)
bert_to_load = bert_name_from_wrapper or DEFAULT_BERT_NAME

st.sidebar.write("Classifier type:", type(clf))
st.sidebar.write("Label encoder present:", label_enc is not None)
st.sidebar.write("Transformer to be used:", bert_to_load)
if embedding_dim:
    st.sidebar.write("Embedding dim (from wrapper):", embedding_dim)

if clf is None:
    st.error("Could not find a classifier inside the wrapper. Ensure wrapper contains 'clf' or 'model'.")
    st.stop()

# ---------- UI ----------
st.write("Paste a news article or claim below and press **Check**.")
text = st.text_area("Article / Claim", height=240, placeholder="Paste the news text here...")

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Loading transformer and predicting..."):
            # load transformer
            try:
                transformer = get_transformer(bert_to_load)
            except Exception as e:
                st.error(f"Failed to load transformer '{bert_to_load}': {e}")
                # show traceback in debug
                st.exception(e)
                st.stop()

            # encode
            try:
                emb = transformer.encode([text.strip()], show_progress_bar=False, convert_to_numpy=True)
                emb = np.asarray(emb)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
            except Exception as e:
                st.error(f"Failed to create embeddings: {e}")
                st.exception(e)
                st.stop()

            # predict
            try:
                raw_pred = clf.predict(emb)[0]
            except Exception as e:
                st.error(f"Classifier predict failed: {e}")
                st.exception(e)
                st.stop()

            # confidence/proba
            conf = safe_proba_from_clf(clf, emb)
            if conf is None:
                conf = 0.5

            label = decode_label(raw_pred, label_enc)

            st.markdown("---")
            st.subheader(f"Prediction: **{label}**")
            st.metric("Confidence", f"{int(round(conf * 100))} %")
            try:
                st.progress(conf)
            except Exception:
                # progress expects 0..1; if broken, ignore
                pass

            if str(label).lower().strip() in {"fake", "0", "false"}:
                st.warning("This content is flagged as likely FAKE. Verify with trusted sources.")
            else:
                st.success("This content is flagged as likely REAL — still verify high-stakes claims.")

# ---------- Debug info ----------
with st.expander("Debug info"):
    st.write("Loaded wrapper type:", type(loaded))
    if isinstance(loaded, dict):
        st.write("Wrapper keys:", list(loaded.keys()))
    if isinstance(loaded, dict) and "__load_error__" in loaded:
        st.text("Wrapper load traceback:")
        st.text(loaded["__load_error__"])
    st.write("Classifier type:", type(clf))
    st.write("Label encoder type:", type(label_enc))
    st.write("Transformer name (to be loaded):", bert_to_load)
    st.write("Working dir:", cwd)
