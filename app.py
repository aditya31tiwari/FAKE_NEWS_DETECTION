# app.py
"""
Streamlit app that runs ONLY on the BERT + XGBoost hybrid model.
Place either:
 - bert_xgb_wrapper.pkl   (recommended: pipeline/wrapper that accepts raw text and implements predict/predict_proba)
OR
 - bert_xgb_model.pkl     (classifier or wrapper saved under this name)

Optional:
 - label_encoder.joblib   (to map numeric labels to strings)

If neither hybrid file exists the app will refuse to run.
"""
import os
import joblib
import numpy as np
import streamlit as st

# ---------- Config: filenames (change if your filenames differ) ----------
HYBRID_WRAPPER = "bert_xgb_wrapper.pkl"
HYBRID_MODEL = "bert_xgb_model.pkl"
LABEL_ENCODER = "label_encoder.joblib"

# ---------- Streamlit page config ----------
st.set_page_config(page_title="DetectoNews — BERT+XGBoost Hybrid", page_icon="🧠", layout="centered")

# ---------- Utilities ----------
@st.cache_resource
def load_artifact(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        # we do not stop app load here; show loader debug later
        return {"__load_error__": str(e)}

def safe_predict_proba(model, X):
    """Return probabilities (1D array) for the 'positive' class if possible, else None."""
    try:
        probs = model.predict_proba(X)
        if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        # handle single-column probability
        return np.asarray(probs).ravel()
    except Exception:
        pass
    try:
        logp = model.predict_log_proba(X)
        logp = np.asarray(logp)
        if logp.ndim == 2 and logp.shape[1] >= 2:
            return np.exp(logp)[:, 1]
        return np.exp(logp).ravel()
    except Exception:
        pass
    try:
        df = model.decision_function(X)
        df = np.asarray(df)
        # if 1d
        if df.ndim == 1:
            return 1.0 / (1.0 + np.exp(-df))
        # if 2d, choose column 1
        if df.ndim == 2 and df.shape[1] >= 2:
            return 1.0 / (1.0 + np.exp(-df[:, 1]))
    except Exception:
        pass
    return None

def predict_with_hybrid(hybrid_obj, texts):
    """
    Try a few ways to get prediction+probability from the hybrid object:
     - assume it accepts raw texts in predict/predict_proba
     - if that fails, try transform(texts) then predict/predict_proba
     - if no proba available, fall back to predict only (confidence 0.5)
    Returns (pred_label_int, confidence_float, used_method_str)
    """
    # normalise input to list
    X_raw = texts if isinstance(texts, (list, tuple)) else [texts]
    # 1) try direct predict_proba on raw texts
    try:
        proba = safe_predict_proba(hybrid_obj, X_raw)
        if proba is not None:
            pred = int(hybrid_obj.predict(X_raw)[0])
            return pred, float(np.clip(proba[0], 0.0, 1.0)), "predict_proba(raw)"
        # if predict_proba returned None but predict works, use predict
        pred = int(hybrid_obj.predict(X_raw)[0])
        return pred, 0.5, "predict(raw) - no proba"
    except Exception:
        pass

    # 2) try using .transform then predict on transformed features
    try:
        if hasattr(hybrid_obj, "transform"):
            X_feats = hybrid_obj.transform(X_raw)
            proba = safe_predict_proba(hybrid_obj, X_feats)
            if proba is not None:
                pred = int(hybrid_obj.predict(X_feats)[0])
                return pred, float(np.clip(proba[0], 0.0, 1.0)), "transform->predict_proba"
            pred = int(hybrid_obj.predict(X_feats)[0])
            return pred, 0.5, "transform->predict (no proba)"
    except Exception:
        pass

    # 3) try calling predict on the object directly (last resort)
    try:
        pred = int(hybrid_obj.predict(X_raw)[0])
        return pred, 0.5, "predict(raw) fallback"
    except Exception:
        pass

    # unknown usage
    raise RuntimeError("Hybrid object doesn't support predict/predict_proba/transform in expected ways.")

def decode_label(pred_int, label_encoder):
    # label_encoder may be a sklearn LabelEncoder or dict-like
    if label_encoder is not None:
        try:
            return str(label_encoder.inverse_transform([pred_int])[0])
        except Exception:
            try:
                return str(label_encoder[pred_int])
            except Exception:
                return str(pred_int)
    return "REAL" if int(pred_int) == 1 else "FAKE"

# ---------- Load artifacts ----------
with st.spinner("Loading hybrid model..."):
    hybrid_wrapper = load_artifact(HYBRID_WRAPPER)
    hybrid_model = None
    if hybrid_wrapper is None:
        hybrid_model = load_artifact(HYBRID_MODEL)

    # If load returned an error dict, convert to None but record error
    hw_error = None
    hm_error = None
    if isinstance(hybrid_wrapper, dict) and "__load_error__" in hybrid_wrapper:
        hw_error = hybrid_wrapper["__load_error__"]
        hybrid_wrapper = None
    if isinstance(hybrid_model, dict) and "__load_error__" in hybrid_model:
        hm_error = hybrid_model["__load_error__"]
        hybrid_model = None

    label_encoder = load_artifact(LABEL_ENCODER)
    if isinstance(label_encoder, dict) and "__load_error__" in label_encoder:
        # ignore label encoder if failed to load
        label_encoder = None

# ensure at least one hybrid artifact exists
if hybrid_wrapper is None and hybrid_model is None:
    st.title("DetectoNews — BERT+XGBoost Hybrid")
    st.error(
        "No hybrid model found. Place 'bert_xgb_wrapper.pkl' (preferred) or 'bert_xgb_model.pkl' in this folder and rerun the app."
    )
    # show load errors for debugging
    with st.expander("Load errors / debug"):
        st.write(f"{HYBRID_WRAPPER} load error: {hw_error}")
        st.write(f"{HYBRID_MODEL} load error: {hm_error}")
    st.stop()

# choose which object to use
hybrid_obj = hybrid_wrapper if hybrid_wrapper is not None else hybrid_model
which_used = HYBRID_WRAPPER if hybrid_wrapper is not None else HYBRID_MODEL

# ---------- UI ----------
st.title("DetectoNews — BERT + XGBoost (Hybrid)")
st.write("This app uses the hybrid BERT → XGBoost model only. Paste a news text/claim below and press Check.")

text = st.text_area("Article / Claim", height=240, placeholder="Paste the news text here...")
if st.button("Check"):
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
            pred_int, confidence, method = predict_with_hybrid(hybrid_obj, [text.strip()])
            label = decode_label(pred_int, label_encoder)

            st.markdown("---")
            st.subheader(f"Prediction: **{label}**")
            st.metric("Confidence", f"{int(round(confidence * 100))} %")
            st.progress(confidence)

            st.info(f"Model used: `{which_used}`  — method: {method}")

            if str(label).lower().strip() in {"fake", "0", "false"}:
                st.warning("This content is flagged as likely FAKE. Verify with trusted sources.")
            else:
                st.success("This content is flagged as likely REAL — still verify high-stakes claims.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
with st.expander("Model debug info"):
    st.write(f"Using: {which_used}")
    if hybrid_wrapper is None and hybrid_model is not None:
        st.write("Note: Only bert_xgb_model.pkl found (no wrapper). The model must accept the features the classifier expects.")
    if hw_error:
        st.write(f"{HYBRID_WRAPPER} load error: {hw_error}")
    if hm_error:
        st.write(f"{HYBRID_MODEL} load error: {hm_error}")
    st.write(f"Label encoder found: {'Yes' if label_encoder is not None else 'No'}")

st.caption("If your hybrid model expects a different interface (e.g., expects precomputed embeddings passed in), save a small wrapper object that exposes .predict and .predict_proba and accepts raw text. I can generate such a wrapper if you want.")
