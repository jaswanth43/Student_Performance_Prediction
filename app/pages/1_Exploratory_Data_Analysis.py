import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")
df = pd.read_csv("dataset/StudentsPerformance_cleaned.csv")
st.title("📊 Exploratory Data Analysis")

st.markdown(
    """
This page presents the exploratory data analysis performed on the
Student Performance dataset before building the machine learning model.
"""
)

st.markdown("---")

st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

st.dataframe(df.head())
st.markdown("---")

st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("**Numerical Features**")
    st.write(df.select_dtypes(exclude="str").columns.tolist())

with col2:
    st.write("**Categorical Features**")
    st.write(df.select_dtypes(include="str").columns.tolist())
st.markdown("---")

st.subheader("Distribution of Student Scores")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

num_features = ["math_score", "reading_score", "writing_score"]

for i, col in enumerate(num_features):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col.replace("_", " ").title())

st.pyplot(fig)
st.markdown("---")

st.subheader("Outlier Analysis")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(num_features):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col.replace("_", " ").title())

st.pyplot(fig)
st.markdown("---")

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(6,5))

sns.heatmap(
    df[num_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig)

st.markdown("---")

st.subheader("Categorical Feature Distribution")

cat_features = [
    "gender",
    "race_ethnicity",
    "parental_education",
    "lunch",
    "test_preparation"
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    sns.countplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(col.replace("_", " ").title())
    axes[i].tick_params(axis="x", rotation=30)

# Remove the unused subplot
fig.delaxes(axes[-1])

plt.tight_layout()
st.pyplot(fig)
st.markdown("---")

st.subheader("Average Scores by Gender")

gender_scores = df.groupby("gender")[[
    "math_score",
    "reading_score",
    "writing_score"
]].mean()

fig, ax = plt.subplots(figsize=(8,5))

gender_scores.plot(kind="bar", ax=ax)

ax.set_ylabel("Average Score")
ax.set_xlabel("Gender")
ax.set_title("Average Scores by Gender")

st.pyplot(fig)

st.markdown("---")

st.subheader("Average Math Score by Test Preparation")

fig, ax = plt.subplots(figsize=(6,4))

sns.barplot(
    data=df,
    x="test_preparation",
    y="math_score",
    ax=ax
)

ax.set_title("Math Score by Test Preparation")

st.pyplot(fig)

st.markdown("---")

st.subheader("📌 Key Insights")

st.success("Reading and writing scores have a strong positive correlation with mathematics scores.")

st.info("Students who completed the test preparation course generally achieved higher mathematics scores.")

st.info("Students with a standard lunch performed better on average than those with free/reduced lunch.")

st.success("Student scores are approximately normally distributed with only a few mild outliers.")

st.warning("The dataset contains both categorical and numerical features, making preprocessing essential before model training.")