import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
with open("model/ridge_model.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📊 Project Information")

st.sidebar.write("### Model")
st.sidebar.success("Ridge Regression")

st.sidebar.write("### Performance")
st.sidebar.metric("R² Score", "0.8806")

st.sidebar.write("### Dataset")
st.sidebar.write("• 1000 Student Records")
st.sidebar.write("• 7 Input Features")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This application predicts a student's mathematics score
using a trained Ridge Regression model.
"""
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🎓 Student Performance Prediction")

st.markdown("""
Welcome to the **Student Performance Prediction System**.

Enter the student's details below to estimate the **Mathematics Score**
using a trained **Machine Learning** model.
""")

st.markdown("---")

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.header("📝 Student Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["female", "male"]
    )

    race_ethnicity = st.selectbox(
        "Race / Ethnicity",
        [
            "group A",
            "group B",
            "group C",
            "group D",
            "group E"
        ]
    )

    parental_education = st.selectbox(
        "Parental Education",
        [
            "associate's degree",
            "bachelor's degree",
            "high school",
            "master's degree",
            "some college",
            "some high school"
        ]
    )

with col2:

    lunch = st.selectbox(
        "Lunch Type",
        [
            "standard",
            "free/reduced"
        ]
    )

    test_preparation = st.selectbox(
        "Test Preparation",
        [
            "completed",
            "none"
        ]
    )

    reading_score = st.slider(
        "Reading Score",
        min_value=0,
        max_value=100,
        value=70
    )

    writing_score = st.slider(
        "Writing Score",
        min_value=0,
        max_value=100,
        value=70
    )

st.markdown("---")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🎯 Predict Mathematics Score", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_education": [parental_education],
        "lunch": [lunch],
        "test_preparation": [test_preparation],
        "reading_score": [reading_score],
        "writing_score": [writing_score]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction Completed Successfully!")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.metric(
            label="🎯 Predicted Mathematics Score",
            value=f"{prediction:.2f}"
        )

    with c2:

        if prediction >= 90:
            st.success("Excellent Performance 🌟")

        elif prediction >= 75:
            st.success("Good Performance 👍")

        elif prediction >= 50:
            st.warning("Average Performance 📘")

        else:
            st.error("Needs Improvement 📚")

    st.markdown("---")

    with st.expander("📄 View Input Details"):
        st.dataframe(input_data, use_container_width=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.caption(
    "Developed by Gummireddy Jaswanth Reddy | Student Performance Prediction using Machine Learning & Streamlit"
)