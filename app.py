import streamlit as st
import joblib 
import numpy as np

import sys
st.write(sys.executable)

# loading the model import joblib
model = joblib.load("logistic_currency_model.pkl")

st.title("💵 Fake Currency Detection System")
st.write("Enter the geometric features of the banknote:")

# Inputs
diagonal = st.number_input("Diagonal", value=0.0)
height_left = st.number_input("Height Left", value=0.0)
height_right = st.number_input("Height Right", value=0.0)
margin_low = st.number_input("Margin Low", value=0.0)
margin_up = st.number_input("Margin Up", value=0.0)
length = st.number_input("Length", value=0.0)

# Prediction
if st.button("Check Currency"):
    sample = np.array([[diagonal, height_left, height_right,
                         margin_low, margin_up, length]])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][prediction]#probabilty of the prediction 

    if prediction == 1:
        st.success(f"✅ Real Currency\nConfidence: {probability*100:.2f}%")
    else:
        st.error(f"❌ Fake Currency\nConfidence: {probability*100:.2f}%")
