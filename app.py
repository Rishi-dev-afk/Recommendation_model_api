import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('course_recommender.pkl')

# App UI
st.title("Course Recommendation System")

# Input fields
age = st.number_input("Age", min_value=15, max_value=50, value=20)
branch = st.selectbox("College Branch", ["CSE", "ECE", "ME", "CE", "EE"])  # modify list as needed
sem = st.selectbox("Semester", ["1", "2", "3", "4", "5", "6", "7", "8"])

# Add more inputs if needed

# Predict
if st.button("Predict Rating"):
    input_df = pd.DataFrame([{
        "age": age,
        "clgbranch": branch,
        "clgsem": sem
        # include other required features
    }])
    prediction = model.predict(input_df)
    st.success(f"Predicted Course Rating: {prediction[0]:.2f}")
