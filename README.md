# Customer Churn Prediction

## 📌 Project Overview

Customer churn prediction is a machine learning project that predicts whether a customer is likely to leave a service.

The goal of this project is to identify customers who have a high probability of churning so that businesses can take proactive retention actions.

## 🎯 Objective

- Analyze customer behavior and churn patterns
- Perform data cleaning and preprocessing
- Create useful features
- Compare multiple machine learning models
- Tune the best-performing model
- Optimize the classification threshold
- Build a customer churn prediction application using Streamlit

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🔄 Project Workflow

1. Data Collection
2. Exploratory Data Analysis
3. Data Cleaning
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Comparison
8. Hyperparameter Tuning
9. Model Evaluation
10. Threshold Optimization
11. Streamlit Deployment

## 🤖 Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression was selected as the final model based on cross-validation ROC-AUC performance.

## 📊 Model Performance

### Final Tuned Logistic Regression

- Best C: 10
- Solver: liblinear
- Best CV ROC-AUC: 0.8466
- Test ROC-AUC: 0.8415
- Test Accuracy: 79.28%

### Classification Threshold

The default threshold of 0.50 was analyzed.

A threshold of **0.35** was selected based on the highest F1-score.

At threshold 0.35:

- Precision: 55.60%
- Recall: 71.66%
- F1-Score: 62.62%

## 💡 Key Business Insights

The analysis identified several customer groups with relatively higher churn:

- Month-to-month contract customers
- New customers with shorter tenure
- Fiber optic internet customers
- Electronic check payment users
- Customers with higher monthly charges

These customer groups can be prioritized for retention campaigns.

## 🌐 Streamlit Application

A Streamlit application was developed where users can enter customer information and receive:

- Churn prediction
- Churn probability
- Customer risk level
- Business retention recommendation

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── threshold.pkl
│   └── model_features.pkl
│
├── notebook/
│   └── customer_churn.ipynb
│
├── app.py
├── requirements.txt
└── README.md
