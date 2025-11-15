# app.py (fixed mapping for label indices)
import os
import joblib
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DetectoNews — BERT+XGBoost", page_icon="🧠")

WRAPPER_FILE = "bert_xgb_wrapper.pkl"

# ----------------------
# Load wrapper
# ----------------------
@st.cache_resource
def load_wrapper():
    return joblib.load(WRAPPER_FILE)

wrapper = load_wrapper()

clf = wrapper["clf"]
label_enc = wrapper["label_encoder"]
bert_name = wrapper.get("bert_name", "all-mpnet-base-v2")

# get classes from label encoder (could be strings or ints)
classes = list(label_enc.classes_)     # e.g. ['FAKE','REAL'] or ['0','1'] or [0,1]

def find_index_for_any(classes_list, candidates):
    """Return first index i where classes_list[i] matches any candidate (with tolerant matching)."""
    for i, c in enumerate(classes_list):
        # direct equality
        if c in candidates:
            return i
        # string-insensitive matching
        try:
            cs = str(c).strip().lower()
            for cand in candidates:
                if str(cand).strip().lower() == cs:
                    return i
        except Exception:
            pass
    return None

# candidates to match for REAL and FAKE
real_candidates = ["REAL", "real", "1", 1, "true", "True"]
fake_candidates = ["FAKE", "fake", "0", 0, "false", "False"]

real_idx = find_index_for_any(classes, real_candidates)
fake_idx = find_index_for_any(classes, fake_candidates)

# fallback: if nothing matched and exactly two classes, assume index 0 = FAKE, 1 = REAL
if real_idx is None or fake_idx is None:
    if len(classes) == 2:
        # assign deterministically but avoid overwriting any found index
        if real_idx is None and fake_idx is None:
            fake_idx, real_idx = 0, 1
        elif real_idx is None:
            # fake_idx known -> other is real
            real_idx = 1 - fake_idx
        elif fake_idx is None:
            fake_idx = 1 - real_idx

# final safety: if still None (weird labels), set both to 0 (so app doesn't crash)
if real_idx is None:
    real_idx = 1 if len(classes) > 1 else 0
if fake_idx is None:
    fake_idx = 0

# Ensure indices are ints and within bounds
n = len(classes)
if not (0 <= real_idx < n and 0 <= fake_idx < n):
    # as last resort, reset to 0/1 if possible
    fake_idx = 0
    real_idx = 1 if n > 1 else 0

# ----------------------
# Load transformer
# ----------------------
@st.cache_resource
def load_transformer(name):
    return SentenceTransformer(name)

model = load_transformer(bert_name)

# ----------------------
# UI
# ----------------------
st.title("DetectoNews — BERT + XGBoost (Fixed Mapping)")
text = st.text_area("Enter article text:", height=250)

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):

            # Embed input
            emb = model.encode([text], convert_to_numpy=True)

            # Predict class index
            pred_idx = clf.predict(emb)[0]

            # Predict probabilities
            proba = clf.predict_proba(emb)[0]

            # Correctly map probabilities using computed indices
            prob_fake = float(proba[fake_idx]) if 0 <= fake_idx < len(proba) else float(proba[0])
            prob_real = float(proba[real_idx]) if 0 <= real_idx < len(proba) else float(proba[-1])

            # Decode final label safely
            if 0 <= pred_idx < len(classes):
                predicted_label = classes[pred_idx]
            else:
                predicted_label = str(pred_idx)

            # Choose final confidence (probability of predicted class)
            final_conf = prob_real if str(predicted_label).strip().lower() in ["real","1","true"] else prob_fake

        # ----------------------
        # Display results
        # ----------------------
        st.subheader(f"Prediction: **{predicted_label}**")
        st.metric("Confidence", f"{final_conf * 100:.1f}%")
        try:
            st.progress(final_conf)
        except Exception:
            pass

        # Interpret predicted_label textually (try to be robust)
        pred_low = str(predicted_label).strip().lower()
        if pred_low in ["fake", "0", "false"]:
            st.warning("This article is likely FAKE. Verify with reliable sources.")
        else:
            st.success("This article seems REAL — but always double-check for accuracy.")

# Debug info
with st.expander("Debug Info"):
    st.write("Transformer name:", bert_name)
    st.write("Label encoder classes (raw):", classes)
    st.write("Interpreted fake_idx:", fake_idx, "real_idx:", real_idx)
    st.write("Predicted class indices must map to these class names.")
