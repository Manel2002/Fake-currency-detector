import streamlit as st
import joblib 
import numpy as np


# loading the model import joblib

try:
    model = joblib.load("logistic_currency_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please check the filename and location.")
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")

st.title("💵 Fake Currency Detection System")
st.write("Enter the geometric features of the banknote:")
# Inputs with placeholders
diagonal = st.number_input("Diagonal", min_value=0.0, format="%.2f")
height_left = st.number_input("Height Left", min_value=0.0, format="%.2f")
height_right = st.number_input("Height Right", min_value=0.0, format="%.2f")
margin_low = st.number_input("Margin Low", min_value=0.0, format="%.2f")
margin_up = st.number_input("Margin Up", min_value=0.0, format="%.2f")
length = st.number_input("Length", min_value=0.0, format="%.2f")

# Prediction only if values are entered
if st.button("Check Currency"):
    if any(v == 0.0 for v in [diagonal, height_left, height_right, margin_low, margin_up, length]):
        st.warning("⚠️ Please enter all values before checking.")
    else:
        sample = np.array([[diagonal, height_left, height_right,
                            margin_low, margin_up, length]])
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][prediction]

        if prediction == 1:
            st.markdown(
                f"<div class='result real'>✅ Real Currency<br>Confidence: {probability*100:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result fake'>❌ Fake Currency<br>Confidence: {probability*100:.2f}%</div>",
                unsafe_allow_html=True
            )
            
#  CSS
st.markdown("""
    <style>
    .result {
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #155724;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #721c24;
    }
    </style>
""", unsafe_allow_html=True)
