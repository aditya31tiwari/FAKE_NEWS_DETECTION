# app.py
"""
Streamlit app for BERT+XGBoost hybrid where the saved wrapper accepts RAW TEXT.
Place bert_xgb_wrapper.pkl (preferred) or bert_xgb_model.pkl in the working directory (or repo root).
Optional: label_encoder.joblib for readable labels.
"""
import os
import joblib
import numpy as np
import streamlit as st
from typing import Optional

# -------- Config (change names if your files differ) --------
WRAPPER_NAME = "bert_xgb_wrapper.pkl"
MODEL_NAME = "bert_xgb_model.pkl"
LABEL_ENCODER_NAME = "label_encoder.joblib"

st.set_page_config(page_title="DetectoNews — BERT+XGBoost", page_icon="🧠", layout="centered")

# -------- Helpers --------
@st.cache_resource
def safe_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        # return dict with error so UI can show it
        return {"__load_error__": str(e)}

def get_proba_from_model(m, texts):
    """
    Try predict_proba -> predict_log_proba -> decision_function (sigmoid) in that order.
    Assumes `m` accepts raw text input (list of strings).
    Returns numpy array or None.
    """
    try:
        p = m.predict_proba(texts)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()
    except Exception:
        pass
    try:
        lp = m.predict_log_proba(texts)
        lp = np.asarray(lp)
        if lp.ndim == 2 and lp.shape[1] >= 2:
            return np.exp(lp)[:, 1]
        return np.exp(lp).ravel()
    except Exception:
        pass
    try:
        df = m.decision_function(texts)
        df = np.asarray(df)
        if df.ndim == 1:
            return 1.0 / (1.0 + np.exp(-df))
        if df.ndim == 2 and df.shape[1] >= 2:
            return 1.0 / (1.0 + np.exp(-df[:, 1]))
    except Exception:
        pass
    return None

def decode_label(pred_int, label_encoder):
    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([pred_int])[0])
        except Exception:
            try:
                return str(label_encoder[pred_int])
            except Exception:
                return str(pred_int)
    return "REAL" if int(pred_int) == 1 else "FAKE"

# -------- Discovery & Load --------
cwd = os.getcwd()
st.sidebar.header("Runtime debug")
st.sidebar.write("Working directory:")
st.sidebar.write(cwd)
try:
    st.sidebar.write("Files:", os.listdir(cwd)[:200])
except Exception:
    st.sidebar.write("Cannot list working dir")

wrapper = safe_load(os.path.join(cwd, WRAPPER_NAME))
model = None
load_errors = {}

# prefer wrapper; otherwise try model file
if wrapper is None:
    # try model name
    model = safe_load(os.path.join(cwd, MODEL_NAME))
else:
    # if wrapper loaded as dict with error, record it and set to None
    if isinstance(wrapper, dict) and "__load_error__" in wrapper:
        load_errors[WRAPPER_NAME] = wrapper["__load_error__"]
        wrapper = None
        model = safe_load(os.path.join(cwd, MODEL_NAME))

# handle model load errors
if model is not None and isinstance(model, dict) and "__load_error__" in model:
    load_errors[MODEL_NAME] = model["__load_error__"]
    model = None

# load label encoder if present
label_enc_obj = safe_load(os.path.join(cwd, LABEL_ENCODER_NAME))
if isinstance(label_enc_obj, dict) and "__load_error__" in label_enc_obj:
    # don't fail app for missing label encoder
    load_errors[LABEL_ENCODER_NAME] = label_enc_obj["__load_error__"]
    label_enc_obj = None

# finalize hybrid object
hybrid_obj = wrapper if wrapper is not None else model
which_path = WRAPPER_NAME if wrapper is not None else (MODEL_NAME if model is not None else None)

# If not found, try one-level down (phase2) and parents (helps some deploys)
if hybrid_obj is None:
    alt_candidates = [
        os.path.join(cwd, "phase2", WRAPPER_NAME),
        os.path.join(cwd, "phase2", MODEL_NAME),
        os.path.join(os.path.dirname(cwd), WRAPPER_NAME),
        os.path.join(os.path.dirname(cwd), MODEL_NAME),
    ]
    for p in alt_candidates:
        if os.path.exists(p):
            loaded = safe_load(p)
            if isinstance(loaded, dict) and "__load_error__" in loaded:
                load_errors[p] = loaded["__load_error__"]
            else:
                hybrid_obj = loaded
                which_path = p
                break

# -------- UI --------
st.title("DetectoNews — BERT + XGBoost Hybrid (raw-text wrapper)")

if hybrid_obj is None:
    st.error("No hybrid wrapper/model found. Place 'bert_xgb_wrapper.pkl' (preferred) or 'bert_xgb_model.pkl' in the working directory.")
    if load_errors:
        with st.expander("Load errors"):
            st.write(load_errors)
    st.stop()

st.sidebar.write("Model loaded from:")
st.sidebar.write(which_path)
if load_errors:
    with st.sidebar.expander("Load warnings/errors"):
        st.write(load_errors)

st.write("This app uses the hybrid wrapper that accepts **raw text**. Paste a news text/claim below and press Check.")
text = st.text_area("Article / Claim", height=260, placeholder="Paste the news text here...")

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            texts = [text.strip()]
            # attempt probability first
            proba = get_proba_from_model(hybrid_obj, texts)
            try:
                pred = int(hybrid_obj.predict(texts)[0])
            except Exception:
                # try transform->predict if wrapper uses transform
                if hasattr(hybrid_obj, "transform"):
                    X = hybrid_obj.transform(texts)
                    pred = int(hybrid_obj.predict(X)[0])
                else:
                    raise

            if proba is None:
                # if predict_proba not available, fallback to 0.5
                confidence = 0.5
            else:
                confidence = float(np.clip(proba[0], 0.0, 1.0))

            label = decode_label(pred, label_enc_obj)

            st.markdown("---")
            st.subheader(f"Prediction: **{label}**")
            st.metric("Confidence", f"{int(round(confidence*100))} %")
            st.progress(confidence)
            st.info(f"Model file: {which_path}")

            if str(label).lower().strip() in {"fake", "0", "false"}:
                st.warning("This content is flagged as likely FAKE. Verify with trusted sources.")
            else:
                st.success("This content is flagged as likely REAL — still verify important claims.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
with st.expander("Debug info"):
    st.write("Working dir:", cwd)
    st.write("Model path used:", which_path)
    st.write("Hybrid object type:", type(hybrid_obj))
    st.write("Label encoder loaded:", label_enc_obj is not None)
    if load_errors:
        st.write("Load warnings/errors:", load_errors)
