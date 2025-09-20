
import streamlit as st
import joblib

#load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#Streamlit App
st.set_page_config(page_title="DetectoNews:A Smart System for Automatic Fake News Detection", page_icon="📰")

st.title("DetectoNews")
st.write("Paste an article and let our model try to decipher wheather its fake or true")

#User Control
user_input = st.text_area("Enter som shi here BRAHH")
if st.button("Check"):
    if user_input.strip() != "":
        try:
            # Preprocess and Predict
            ip_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(ip_vectorized)[0]

            if prediction == 1:
                st.success("✅ Yeah! It's Alright Brahh (Looks REAL)")
            else:
                st.error("⚠️ Nahh Man! This looks FAKE dude!!")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
'''
import streamlit as st

st.title("Hello Streamlit 👋")
st.write("If you can see this, your Streamlit setup is working fine!")

name = st.text_input("Enter your name:")
if st.button("Greet"):
    st.success(f"Hello, {name}! 🎉")
'''
