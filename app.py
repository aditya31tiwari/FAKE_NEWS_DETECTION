
import streamlit as st
import joblib

#load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#Streamlit App
st.set_page_config(page_title="DetectoNews:A Smart System for Automatic Fake News Detection", page_icon="🕵️")

st.title("DetectoNews")
st.write("Paste an article and let our model try to decipher wheather its fake or true")

#User Control
user_input = st.text_area("Enter Here:")
if st.button("Check"):
    if user_input.strip() != "":
        try:
            # Preprocess and Predict
            ip_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(ip_vectorized)[0]

            if prediction == 1:
                st.success("Yeah! It's Alright(Looks REAL)")
            else:
                st.error("This looks FAKE!!")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

'''This is it, This is where the magic happens '''
