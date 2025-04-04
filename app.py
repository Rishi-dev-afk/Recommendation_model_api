import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('course_recommender.pkl')

st.title("Course Rating Predictor")

# Inputs
video_title = st.text_input("Video Title")
clgyear = st.selectbox("College Year", ["First", "Second", "Third", "Fourth"])
clgsem = st.number_input("Semester", min_value=1, max_value=8)

if st.button("Predict Rating"):
    # Match the exact column names your model expects
    input_df = pd.DataFrame([{
        "video_title": video_title,
        "clgyear": clgyear,
        "clgsem": clgsem
    }])

    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Course Rating: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
