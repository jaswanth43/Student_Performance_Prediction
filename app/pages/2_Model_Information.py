import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Model Information",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Machine Learning Model")

st.markdown("""
This page presents the machine learning models evaluated for predicting
students' mathematics scores and highlights the performance of the selected model.
""")

st.markdown("---")
results_df = pd.DataFrame({
    "Model":[
        "Ridge Regression",
        "Linear Regression",
        "Gradient Boosting",
        "AdaBoost",
        "Random Forest",
        "Lasso",
        "Decision Tree"
    ],

    "R2 Score":[
        0.8806,
        0.8804,
        0.8722,
        0.8516,
        0.8504,
        0.8254,
        0.7473
    ],

    "RMSE":[
        5.39,
        5.39,
        5.58,
        6.01,
        6.03,
        6.52,
        7.84
    ]
})

st.subheader("Model Comparison")

st.dataframe(results_df, use_container_width=True)

st.markdown("---")

st.subheader("Model Performance Comparison")

fig, ax = plt.subplots(figsize=(9,5))

sns.barplot(
    data=results_df,
    x="R2 Score",
    y="Model",
    ax=ax
)

ax.set_title("Comparison of Regression Models")

st.pyplot(fig)


st.markdown("---")

st.subheader("🏆 Selected Model")

st.success("""
**Ridge Regression** was selected as the final model because it achieved:

- Highest R² Score (0.8806)
- Lowest RMSE (5.39)
- Lowest MAE (4.21)

It also includes L2 Regularization, which improves generalization and helps reduce overfitting.
""")

st.markdown("---")

st.subheader("Performance Metrics")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("R² Score", "0.8806")

with c2:
    st.metric("MAE", "4.21")

with c3:
    st.metric("RMSE", "5.39")


st.markdown("---")

st.subheader("Machine Learning Workflow")

st.info("""
1. Data Cleaning & Preprocessing

2. Exploratory Data Analysis (EDA)

3. Feature Selection

4. Train-Test Split

5. Data Preprocessing (Scaling & Encoding)

6. Model Training

7. Model Comparison

8. Final Model Selection

9. Prediction & Deployment
""")

st.markdown("---")

st.subheader("Conclusion")

st.success("""
The Ridge Regression model explains approximately **88% of the variation**
in students' mathematics scores.

The model provides reliable predictions with low prediction error and
was deployed as an interactive Streamlit web application.
""")