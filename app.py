import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Load saved files
# --------------------------------------------------

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")
features = joblib.load("models/model_features.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to predict the probability of churn."
)


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# --------------------------------------------------
# Billing Information
# --------------------------------------------------

st.header("Billing Information")

col1, col2, col3 = st.columns(3)


with col1:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )


col1, col2, col3 = st.columns(3)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Churn"):

    # AvgChargePerTenure
    if tenure > 0:
        avg_charge = total_charges / tenure
    else:
        avg_charge = monthly_charges


    # TenureGroup
    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Medium"
    elif tenure <= 48:
        tenure_group = "Loyal"
    else:
        tenure_group = "Long-term"


    # ChargeGroup
    if monthly_charges <= 40:
        charge_group = "Low"
    elif monthly_charges <= 70:
        charge_group = "Medium"
    elif monthly_charges <= 100:
        charge_group = "High"
    else:
        charge_group = "Very High"


    # --------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------

    input_data = pd.DataFrame({
        "SeniorCitizen": [
            1 if senior_citizen == "Yes" else 0
        ],

        "tenure": [tenure],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges],

        "AvgChargePerTenure": [avg_charge],

        "gender_Male": [
            1 if gender == "Male" else 0
        ],

        "Partner_Yes": [
            1 if partner == "Yes" else 0
        ],

        "Dependents_Yes": [
            1 if dependents == "Yes" else 0
        ],

        "PhoneService_Yes": [
            1 if phone_service == "Yes" else 0
        ],

        "MultipleLines_No phone service": [
            1 if multiple_lines == "No phone service" else 0
        ],

        "MultipleLines_Yes": [
            1 if multiple_lines == "Yes" else 0
        ],

        "InternetService_Fiber optic": [
            1 if internet_service == "Fiber optic" else 0
        ],

        "InternetService_No": [
            1 if internet_service == "No" else 0
        ],

        "OnlineSecurity_No internet service": [
            1 if online_security == "No internet service" else 0
        ],

        "OnlineSecurity_Yes": [
            1 if online_security == "Yes" else 0
        ],

        "OnlineBackup_No internet service": [
            1 if online_backup == "No internet service" else 0
        ],

        "OnlineBackup_Yes": [
            1 if online_backup == "Yes" else 0
        ],

        "DeviceProtection_No internet service": [
            1 if device_protection == "No internet service" else 0
        ],

        "DeviceProtection_Yes": [
            1 if device_protection == "Yes" else 0
        ],

        "TechSupport_No internet service": [
            1 if tech_support == "No internet service" else 0
        ],

        "TechSupport_Yes": [
            1 if tech_support == "Yes" else 0
        ],

        "StreamingTV_No internet service": [
            1 if streaming_tv == "No internet service" else 0
        ],

        "StreamingTV_Yes": [
            1 if streaming_tv == "Yes" else 0
        ],

        "StreamingMovies_No internet service": [
            1 if streaming_movies == "No internet service" else 0
        ],

        "StreamingMovies_Yes": [
            1 if streaming_movies == "Yes" else 0
        ],

        "Contract_One year": [
            1 if contract == "One year" else 0
        ],

        "Contract_Two year": [
            1 if contract == "Two year" else 0
        ],

        "PaperlessBilling_Yes": [
            1 if paperless_billing == "Yes" else 0
        ],

        "PaymentMethod_Credit card (automatic)": [
            1 if payment_method == "Credit card (automatic)" else 0
        ],

        "PaymentMethod_Electronic check": [
            1 if payment_method == "Electronic check" else 0
        ],

        "PaymentMethod_Mailed check": [
            1 if payment_method == "Mailed check" else 0
        ],

        "TenureGroup_Medium": [
            1 if tenure_group == "Medium" else 0
        ],

        "TenureGroup_Loyal": [
            1 if tenure_group == "Loyal" else 0
        ],

        "TenureGroup_Long-term": [
            1 if tenure_group == "Long-term" else 0
        ],

        "ChargeGroup_Medium": [
            1 if charge_group == "Medium" else 0
        ],

        "ChargeGroup_High": [
            1 if charge_group == "High" else 0
        ],

        "ChargeGroup_Very High": [
            1 if charge_group == "Very High" else 0
        ]
    })


    # Ensure exact feature order
    input_data = input_data[features]


    # Scale
    input_scaled = scaler.transform(input_data)


    # Probability
    churn_probability = model.predict_proba(
        input_scaled
    )[0][1]


    # Prediction using threshold 0.35
    prediction = (
        churn_probability >= threshold
    )


    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    st.header("Prediction Result")

    probability_percent = churn_probability * 100

   # --------------------------------------------------
# Prediction Result
# --------------------------------------------------

st.header("Prediction Result")

probability_percent = churn_probability * 100

# Risk level
if probability_percent >= 70:
    risk_level = "High Risk"
elif probability_percent >= 40:
    risk_level = "Medium Risk"
else:
    risk_level = "Low Risk"


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Churn Probability",
        f"{probability_percent:.2f}%"
    )

with col2:
    st.metric(
        "Risk Level",
        risk_level
    )

with col3:
    st.metric(
        "Classification Threshold",
        f"{threshold:.2f}"
    )


# Prediction message

if prediction:

    st.error(
        "🔴 Customer is likely to CHURN"
    )

    st.subheader("💡 Recommended Business Action")

    st.write(
        "Consider contacting this customer with a suitable "
        "retention offer, personalized support, or a plan upgrade."
    )

else:

    st.success(
        "🟢 Customer is unlikely to CHURN"
    )

    st.subheader("💡 Recommended Business Action")

    st.write(
        "Continue providing good service and monitor customer "
        "satisfaction to maintain loyalty."
    )


# Probability bar

st.subheader("Churn Probability")

st.progress(
    min(churn_probability, 1.0)
)