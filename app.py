# app.py
"""
Streamlit app for BERT+XGBoost hybrid (lightweight wrapper + runtime BERT).
Expect a small wrapper file (or XGBoost + metadata dict) named:
 - bert_xgb_wrapper.pkl (preferred)
or
 - bert_xgb_wrapper_light.pkl (alternate)
Optional label encoder: label_encoder.joblib

The app will lazy-load the SentenceTransformer at first prediction.
"""
import os
import joblib
import numpy as np
import streamlit as st
from typing import Optional

# ---------------- Config ----------------
WRAPPER_FILES = ["bert_xgb_wrapper.pkl", "bert_xgb_wrapper_light.pkl", "bert_xgb_model.pkl"]
LABEL_ENCODER = "label_encoder.joblib"
# default HF model name you used when training; change if you used a different one
DEFAULT_BERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="DetectoNews — BERT+XGBoost", page_icon="🧠", layout="centered")

# ---------------- Utilities ----------------
@st.cache_resource
def load_joblib(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        return {"__load_error__": str(e)}

@st.cache_resource
def get_sentence_transformer(name: str):
    # Heavy object cached by Streamlit
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

def make_light_wrapper_from_dict(d: dict, bert_name_fallback: str = DEFAULT_BERT_NAME):
    """
    Convert a dict (saved in .pkl) into a small wrapper object with .predict and .predict_proba.
    Acceptable keys (various naming):
      - 'model' or 'clf' -> classifier (XGBoost)
      - 'bert_name' -> HF model name (preferred)
      - 'bert' -> a loaded SentenceTransformer object (we will ignore embedded model and load runtime one)
      - 'label_encoder' or 'le' -> label encoder object (optional)
    """
    clf = None
    le = None
    bert_name = bert_name_fallback

    if "model" in d:
        clf = d["model"]
    elif "clf" in d:
        clf = d["clf"]
    elif "xgb" in d:
        clf = d["xgb"]
    elif "classifier" in d:
        clf = d["classifier"]

    if "label_encoder" in d:
        le = d["label_encoder"]
    elif "le" in d:
        le = d["le"]

    if "bert_name" in d:
        bert_name = d["bert_name"]
    elif "transformer_name" in d:
        bert_name = d["transformer_name"]

    # if the dict accidentally contains 'bert' (a heavy object), ignore it; we'll load at runtime
    return LightweightHybridWrapper(clf, label_encoder=le, bert_name=bert_name)

class LightweightHybridWrapper:
    """
    Small wrapper that stores XGBoost classifier + label encoder + name of bert model.
    It lazy-loads SentenceTransformer during transform/predict calls.
    """
    def __init__(self, clf, label_encoder=None, bert_name: str = DEFAULT_BERT_NAME):
        self.clf = clf
        self.le = label_encoder
        self.bert_name = bert_name
        self._bert = None  # will be set on first use (not pickled if you save this object after removing bert)

    def _ensure_bert(self):
        if self._bert is None:
            # use cached loader
            self._bert = get_sentence_transformer(self.bert_name)

    def transform(self, texts):
        self._ensure_bert()
        # disable progress bar inside streamlit
        return self._bert.encode(texts, show_progress_bar=False)

    def predict(self, texts):
        X = self.transform(texts)
        preds = self.clf.predict(X)
        # if label encoder is present, return decoded labels, otherwise raw preds
        if self.le is not None:
            try:
                return self.le.inverse_transform(preds)
            except Exception:
                return preds
        return preds

    def predict_proba(self, texts):
        X = self.transform(texts)
        return self.clf.predict_proba(X)

# ---------------- Discovery & Load ----------------
cwd = os.getcwd()
st.sidebar.header("Runtime diagnostics")
st.sidebar.write("Working dir:")
st.sidebar.write(cwd)
try:
    st.sidebar.write(os.listdir(cwd)[:200])
except Exception:
    st.sidebar.write("Cannot list working directory")

loaded_wrapper = None
wrapper_path_used = None
load_warnings = {}

# try candidate files
for fname in WRAPPER_FILES:
    if os.path.exists(os.path.join(cwd, fname)):
        obj = load_joblib(os.path.join(cwd, fname))
        if obj is None:
            continue
        if isinstance(obj, dict) and "__load_error__" in obj:
            load_warnings[fname] = obj["__load_error__"]
            continue

        # if it's already a wrapper-like (has predict method), use as-is
        if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
            loaded_wrapper = obj
            wrapper_path_used = os.path.join(cwd, fname)
            break

        # if it's a dict, try to build a lightweight wrapper from it
        if isinstance(obj, dict):
            try:
                loaded_wrapper = make_light_wrapper_from_dict(obj)
                wrapper_path_used = os.path.join(cwd, fname) + " (dict->wrapper)"
                break
            except Exception as e:
                load_warnings[fname] = f"dict->wrapper conversion failed: {e}"
                continue

        # if it's a classifier (e.g., xgboost object) then wrap it
        # Heuristics: xgboost.XGBClassifier or has predict_proba attribute
        if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
            # treat this as clf-only: create small wrapper with default bert name
            loaded_wrapper = LightweightHybridWrapper(obj, label_encoder=None, bert_name=DEFAULT_BERT_NAME)
            wrapper_path_used = os.path.join(cwd, fname) + " (clf-only)"
            break

        # otherwise unknown object: show preview
        load_warnings[fname] = f"Loaded object type {type(obj)} is not wrapper/ dict/ clf."

# try to find label encoder separately (optional)
label_enc = None
if os.path.exists(os.path.join(cwd, LABEL_ENCODER)):
    le_obj = load_joblib(os.path.join(cwd, LABEL_ENCODER))
    if isinstance(le_obj, dict) and "__load_error__" in le_obj:
        load_warnings[LABEL_ENCODER] = le_obj["__load_error__"]
    else:
        label_enc = le_obj

# if we converted a wrapper from dict but no label encoder set, try to attach from dict
if label_enc is None and loaded_wrapper is None:
    # nothing we can do
    pass

# If loaded_wrapper is a LightweightHybridWrapper but label_enc exists, attach
if isinstance(loaded_wrapper, LightweightHybridWrapper) and label_enc is not None and loaded_wrapper.le is None:
    loaded_wrapper.le = label_enc

# If still nothing
if loaded_wrapper is None:
    st.title("DetectoNews — BERT+XGBoost")
    st.error("No usable hybrid wrapper found. Place a lightweight wrapper (or clf/dict) in the working directory with one of these names: " + ", ".join(WRAPPER_FILES))
    if load_warnings:
        with st.expander("Load warnings/errors"):
            st.write(load_warnings)
    st.stop()

# ---------------- UI ----------------
st.title("DetectoNews — BERT + XGBoost (runtime BERT loader)")
st.write("Model loaded from: " + str(wrapper_path_used))
if load_warnings:
    with st.expander("Load warnings"):
        st.write(load_warnings)

st.write("Paste news text/claim below and press Check (first call will load BERT and take longer).")
text = st.text_area("Article / Claim", height=240, placeholder="Paste the news text here...")

if st.button("Check"):
    if not text.strip():
        st.warning("Enter some text first.")
        st.stop()

    try:
        # ensure the wrapper has a bert name; if a plain clf was used we use default bert name
        if isinstance(loaded_wrapper, LightweightHybridWrapper):
            # lazy-load happens inside predict
            pred_obj = loaded_wrapper.predict([text.strip()])
            try:
                # predict might return array-like
                pred = list(pred_obj)[0]
            except Exception:
                pred = pred_obj
            # get proba
            try:
                proba_raw = loaded_wrapper.predict_proba([text.strip()])
                proba = float(np.clip(np.asarray(proba_raw)[0, 1], 0.0, 1.0))
            except Exception:
                proba = 0.5

        else:
            # generic wrapper-like object
            pred_obj = loaded_wrapper.predict([text.strip()])
            try:
                pred = list(pred_obj)[0]
            except Exception:
                pred = pred_obj
            proba = 0.5
            try:
                pr = loaded_wrapper.predict_proba([text.strip()])
                proba = float(np.clip(np.asarray(pr)[0, 1], 0.0, 1.0))
            except Exception:
                pass

        # decode if label encoder present
        try:
            if hasattr(loaded_wrapper, "le") and loaded_wrapper.le is not None:
                try:
                    pred_display = loaded_wrapper.le.inverse_transform([pred])[0]
                except Exception:
                    pred_display = str(pred)
            else:
                pred_display = "REAL" if int(pred) == 1 else "FAKE"
        except Exception:
            pred_display = str(pred)

        st.markdown("---")
        st.subheader(f"Prediction: **{pred_display}**")
        st.metric("Confidence", f"{int(round(proba * 100))} %")
        st.progress(proba)
        if str(pred_display).lower().strip() in {"fake", "0", "false"}:
            st.warning("This content is flagged as likely FAKE. Verify with trusted sources.")
        else:
            st.success("This content is flagged as likely REAL — still verify high-stakes claims.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
with st.expander("More debug info"):
    st.write("Wrapper object type:", type(loaded_wrapper))
    st.write("Wrapper path used:", wrapper_path_used)
    st.write("Label encoder loaded:", label_enc is not None)
    st.write("Working dir:", cwd)
