# app.py
import os
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="DetectoNews: Hybrid Detector", page_icon="🕵️", layout="centered")

# ---------- Config - filenames present in your repo ----------
HYBRID_WRAPPER = "bert_xgb_wrapper.pkl"   # likely a wrapper/pipeline (preferred)
HYBRID_MODEL = "bert_xgb_model.pkl"       # maybe classifier alone
NB_MODEL = "naive_bayes_model.pkl"        # your older fallback
TFIDF = "tfidf_vectorizer.pkl"
LABEL_ENCODER = "label_encoder.joblib"

# ---------- Helpers ----------
@st.cache_resource
def load_artifact(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

def safe_predict_proba(model, X):
    """Try predict_proba, predict_log_proba (exp), or decision_function -> sigmoid"""
    try:
        probs = model.predict_proba(X)
        # if multiclass with >=2 columns, assume index 1 is 'positive/fake' label
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        # single column (rare)
        return probs.ravel()
    except Exception:
        pass
    try:
        logp = model.predict_log_proba(X)
        if logp.ndim == 2 and logp.shape[1] >= 2:
            return np.exp(logp)[:, 1]
        return np.exp(logp).ravel()
    except Exception:
        pass
    try:
        df = model.decision_function(X)
        # if 1d array -> map via sigmoid
        if isinstance(df, np.ndarray):
            if df.ndim == 1:
                return 1.0 / (1.0 + np.exp(-df))
            # if multiclass decision_function, try to pick column 1
            if df.ndim == 2 and df.shape[1] >= 2:
                return 1.0 / (1.0 + np.exp(-df[:, 1]))
    except Exception:
        pass
    return None

def get_label_from_pred(pred, label_encoder):
    """Return readable label for numeric prediction, falling back to int/string."""
    if label_encoder is not None:
        try:
            # label_encoder could be sklearn LabelEncoder or similar
            return str(label_encoder.inverse_transform([pred])[0])
        except Exception:
            try:
                # if it's a dict-like mapping
                return str(label_encoder[pred])
            except Exception:
                return str(pred)
    # default mapping common in your old app: 1 -> REAL, 0 -> FAKE (but verify)
    return "REAL" if int(pred) == 1 else "FAKE"

# ---------- Load models ----------
with st.spinner("Loading models..."):
    hybrid_wrapper = load_artifact(HYBRID_WRAPPER)
    hybrid_model = load_artifact(HYBRID_MODEL) if hybrid_wrapper is None else None
    nb_model = load_artifact(NB_MODEL) if hybrid_wrapper is None and hybrid_model is None else None
    tfidf = load_artifact(TFIDF) if nb_model is not None else None
    label_encoder = load_artifact(LABEL_ENCODER)

# ---------- UI ----------
st.title("DetectoNews")
st.write("Paste an article/claim below and the model will predict REAL vs FAKE and show a confidence score.")

text = st.text_area("Article / Claim", height=220, placeholder="Paste the news text here...")
if st.button("Check"):
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        input_texts = [text.strip()]
        try:
            used = None
            proba = None
            pred = None

            # 1) If wrapper pipeline present, assume it accepts raw text (or has a transform inside)
            if hybrid_wrapper is not None:
                used = os.path.basename(HYBRID_WRAPPER)
                # Some pipelines accept raw text, others expect embeddings; try both.
                try:
                    proba = safe_predict_proba(hybrid_wrapper, input_texts)
                    if proba is not None:
                        pred = int(hybrid_wrapper.predict(input_texts)[0])
                    else:
                        pred = int(hybrid_wrapper.predict(input_texts)[0])
                        proba = safe_predict_proba(hybrid_wrapper, input_texts)  # try again
                except Exception:
                    # maybe wrapper expects embeddings -> check if it has a method to transform text
                    if hasattr(hybrid_wrapper, "transform"):
                        X = hybrid_wrapper.transform(input_texts)
                        proba = safe_predict_proba(hybrid_wrapper, X)
                        pred = int(hybrid_wrapper.predict(X)[0])
                    else:
                        # try to call predict on input_texts and accept no confidence
                        pred = int(hybrid_wrapper.predict(input_texts)[0])

            # 2) If there is a standalone hybrid model (maybe accepts features), try it
            elif hybrid_model is not None:
                used = os.path.basename(HYBRID_MODEL)
                # try direct on raw text (some wrappers saved under different name)
                try:
                    proba = safe_predict_proba(hybrid_model, input_texts)
                    pred = int(hybrid_model.predict(input_texts)[0])
                except Exception:
                    # fallback: try to transform with tfidf/embedder if present
                    # If tfidf exists, use it
                    if tfidf is not None:
                        X = tfidf.transform(input_texts)
                        proba = safe_predict_proba(hybrid_model, X)
                        pred = int(hybrid_model.predict(X)[0]) if proba is None else int(hybrid_model.predict(X)[0])
                    else:
                        # final fallback: try predict on raw
                        pred = int(hybrid_model.predict(input_texts)[0])

            # 3) fallback to Naive Bayes + tfidf (your original app)
            elif nb_model is not None and tfidf is not None:
                used = "NaiveBayes + TFIDF (fallback)"
                X = tfidf.transform(input_texts)
                proba = safe_predict_proba(nb_model, X)
                pred = int(nb_model.predict(X)[0])
            else:
                st.error("No usable model found. Place bert_xgb_wrapper.pkl or bert_xgb_model.pkl or naive_bayes_model.pkl + tfidf_vectorizer.pkl in this folder.")
                st.stop()

            # compute confidence
            if proba is None:
                # No probability available, set neutral 50%
                confidence = 0.5
            else:
                # proba may be array-like
                confidence = float(np.clip(proba[0], 0.0, 1.0)) if hasattr(proba, "__len__") else float(np.clip(proba, 0.0, 1.0))

            label_text = get_label_from_pred(pred, label_encoder)

            # UI results
            st.markdown("---")
            st.subheader(f"Prediction: **{label_text}**")
            st.metric(label="Confidence", value=f"{int(round(confidence*100))} %")
            st.progress(confidence)

            if label_text.lower().strip() in ["fake", "0", "false"]:
                st.warning("This content is flagged as likely FAKE. Verify with trusted sources.")
            else:
                st.success("This content is flagged as likely REAL, but still verify high-stakes claims.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
with st.expander("Model files detected / debug"):
    st.write(f"{HYBRID_WRAPPER}: {'Yes' if hybrid_wrapper is not None else 'No'}")
    st.write(f"{HYBRID_MODEL}: {'Yes' if hybrid_model is not None else 'No'}")
    st.write(f"{NB_MODEL}: {'Yes' if nb_model is not None else 'No'}")
    st.write(f"{TFIDF}: {'Yes' if tfidf is not None else 'No'}")
    st.write(f"{LABEL_ENCODER}: {'Yes' if label_encoder is not None else 'No'}")

st.caption("Notes: If your wrapper/pipeline expects embeddings rather than raw text, ensure the wrapper object exposes a `.transform(list_of_texts)` and `.predict`/`.predict_proba` methods. If your saved models use different names, update the constants at the top.")
