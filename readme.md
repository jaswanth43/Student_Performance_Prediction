# 🎓 Student Performance Prediction using Machine Learning
## 🌐 Live Demo

**Streamlit Application:**

https://studentperformanceprediction-doyflwb9gr3qwiady8y2mf.streamlit.app/

## 📌 Project Overview

This project predicts students' **Mathematics Scores** using demographic and academic information. The complete machine learning workflow includes data cleaning, exploratory data analysis (EDA), model training, evaluation, and deployment through an interactive Streamlit web application.

The objective is to identify the most suitable regression model for predicting student performance and provide an easy-to-use interface for real-time predictions.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Multiple Regression Model Comparison
- Ridge Regression Model Selection
- Interactive Streamlit Web Application
- Real-time Mathematics Score Prediction

---

## 📂 Project Structure

```
Student_Performance_Prediction/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Exploratory_Data_Analysis.py
│       └── 2_Model_Information.py
│
├── dataset/
│   ├── StudentsPerformance.csv
│   └── StudentsPerformance_cleaned.csv
│
├── model/
│   └── ridge_model.pkl
│
├── notebooks/
│   ├── 01_Data_Cleaning_Preprocessing.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   └── 03_Model_Training_Evaluation.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset Information

- **Dataset Name:** Students Performance Dataset
- **Total Records:** 1000
- **Features:** 8

### Input Features

- Gender
- Race / Ethnicity
- Parental Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

### Target Variable

- Mathematics Score

---

## 📈 Exploratory Data Analysis

The following analyses were performed:

- Dataset Overview
- Missing Value Analysis
- Duplicate Record Analysis
- Numerical Feature Distribution
- Outlier Detection
- Categorical Feature Distribution
- Correlation Heatmap
- Pairplot Analysis
- Feature vs Target Analysis
- Group-wise Analysis

### Key Insights

- Reading and writing scores have a strong positive correlation with mathematics scores.
- Students completing the test preparation course generally achieved higher scores.
- Students with standard lunch performed better on average.
- Student scores are approximately normally distributed.
- Only a few mild outliers were observed.

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor

### Model Performance

| Model | R² Score |
|--------|---------:|
| Ridge Regression | **0.8806** |
| Linear Regression | 0.8804 |
| Gradient Boosting | 0.8722 |
| AdaBoost | 0.8516 |
| Random Forest | 0.8504 |
| Lasso Regression | 0.8254 |
| Decision Tree | 0.7473 |

### Final Model

**Ridge Regression** was selected because it achieved:

- Highest R² Score
- Lowest MAE
- Lowest RMSE
- Better generalization through L2 Regularization

---

## 📊 Final Evaluation

- **R² Score:** 0.8806
- **MAE:** 4.21
- **RMSE:** 5.39

---

## 💻 Streamlit Application

The project includes an interactive Streamlit dashboard with:

### 🏠 Home
- Student information form
- Real-time mathematics score prediction

### 📊 Exploratory Data Analysis
- Dataset overview
- Distribution plots
- Correlation heatmap
- Outlier analysis
- Key insights

### 📈 Model Information
- Model comparison
- Performance metrics
- Final model selection
- Machine learning workflow

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pickle

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/jaswanth43/Student_Performance_Prediction.git
```

Navigate to the project directory:

```bash
cd Student_Performance_Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/Home.py
```

---

## 🎯 Future Enhancements

- Hyperparameter Optimization
- Feature Engineering
- Additional Regression Models
- Model Deployment on Cloud
- Performance Monitoring Dashboard

---

## 👨‍💻 Author

**Gummireddy Jaswanth Reddy**

If you found this project useful, feel free to ⭐ the repository.
