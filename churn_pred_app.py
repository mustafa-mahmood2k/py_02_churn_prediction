# Gender -> 1 Female    0 Male
# Churn -> 1 Yes    0 No
# ContractType -> one-hot encoded (Month-to-month / One year / Two year)
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Column names/order are exported as feature_columns.pkl - loaded below instead of
# hardcoded, so this app can't drift out of sync with however the model was actually trained.

import streamlit as st
import joblib
import pandas as pd

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter values and press predict to get a prediction")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter tenure", min_value=0,
                         max_value=130, value=10)  # value = base value

monthlycharge = st.number_input(
    "Enter monthly charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the gender", ["Male", "Female"])

contract_type = st.selectbox(
    "Enter the contract type", ["Month-to-month", "One year", "Two year"])

st.divider()

predict_button = st.button("Predict")

if predict_button:
    gender_selected = 1 if gender == "Female" else 0

    # Build a one-row DataFrame the same way the notebook builds X, then one-hot encode
    # ContractType the same way pd.get_dummies() did during training.
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender_selected,
        "Tenure": tenure,
        "MonthlyCharges": monthlycharge,
        "ContractType": contract_type,
    }])
    input_df = pd.get_dummies(input_df, columns=["ContractType"])

    # Align to the exact columns/order the model was trained on. Any dummy column not
    # produced above (e.g. the dropped baseline category) gets filled with 0.
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    X_array = scaler.transform(input_df)
    prediction = model.predict(X_array)[0]
    predicted = "Churn" if prediction == 1 else "Not Churn"
    st.write(f"Predicted: {predicted}")
else:
    st.write("Please enter value and use predict button")
