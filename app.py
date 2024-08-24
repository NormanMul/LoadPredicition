import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    return model

model = load_model()

# Streamlit app
st.title('Loan Quality Prediction App')

# Sidebar
st.sidebar.title("Loan Prediction Model")

st.header("Upload the Excel File")

# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file)
    st.write("Input Data:")
    st.write(input_df.head())

    # Assuming the input data has already been preprocessed accordingly
    if st.button('Predict Loan Quality'):
        # Predicting loan quality using the pre-trained model
        predictions = model.predict(input_df)
        st.write("Predicted Loan Quality/Outcome:")
        st.write(predictions)
