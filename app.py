# app.py
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

classes = list(label_enc.classes_)     # e.g. ['FAKE', 'REAL']
fake_idx = classes.index("FAKE")
real_idx = classes.index("REAL")


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
st.title("DetectoNews — BERT + XGBoost (Fixed Version)")
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

            # Correctly map probabilities using label encoder
            prob_fake = proba[fake_idx]
            prob_real = proba[real_idx]

            # Decode final label
            predicted_label = classes[pred_idx]

            # Final confidence = prob of the predicted class
            final_conf = prob_real if predicted_label == "REAL" else prob_fake

        # ----------------------
        # Display results
        # ----------------------
        st.subheader(f"Prediction: **{predicted_label}**")
        st.metric("Confidence", f"{final_conf * 100:.1f}%")
        st.progress(final_conf)

        if predicted_label == "FAKE":
            st.warning("This article is likely FAKE. Verify with reliable sources.")
        else:
            st.success("This article seems REAL — but always double-check for accuracy.")

# Debug info
with st.expander("Debug Info"):
    st.write("Transformer name:", bert_name)
    st.write("Classes:", classes)
    st.write("Fake index:", fake_idx)
    st.write("Real index:", real_idx)
