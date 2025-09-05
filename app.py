import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load

# Load model, scaler, and medians
@st.cache_resource
def load_resources():
    try:
        model = load("models/diabetes.sav")
        scaler = load("models/scaler.sav")
        medians = load("models/medians.sav")
        return model, scaler, medians
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        return None, None, None

# Input validation
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors = []
    warnings = []
    if glucose <= 0:
        errors.append("Glucose level must be greater than 0")
    if bloodpressure <= 0:
        errors.append("Blood pressure must be greater than 0")
    if bmi <= 0:
        errors.append("BMI must be greater than 0")
    if age <= 0:
        errors.append("Age must be greater than 0")
    if pregnancies > 0 and age < 12:
        errors.append("Age seems too low for number of pregnancies")
    return errors, warnings

# Preprocess input
def preprocess_input(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age, scaler, medians):
    try:
        input_dict = {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [bloodpressure],
            "SkinThickness": [skinthickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetespedigree],
            "Age": [age],
        }
        df = pd.DataFrame(input_dict)

        # Replace zeros with NaN for selected cols
        cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

        # Fill NaN with training medians
        df = df.fillna(medians)

        # Scale
        if scaler is not None:
            input_scaled = scaler.transform(df)
        else:
            input_scaled = df.values

        return input_scaled
    except Exception as e:
        st.error(f"Error preprocessing input data: {str(e)}")
        return None

# Prediction
def predict_diabetes(model, scaler, medians, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    input_data = preprocess_input(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age, scaler, medians)
    if input_data is None:
        return None, None
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] * 100
    return bool(prediction[0]), prob

# Display results
def display_results(name, prediction, probability, errors, warnings):
    if prediction is None:
        st.error("❌ Could not make prediction.")
        return
    if prediction:
        st.error(f"🔴 Hello {name}, risk assessment: **POSITIVE**")
        st.error(f"**Risk Probability: {probability:.1f}%**")
    else:
        st.success(f"✅ Hello {name}, risk assessment: **NEGATIVE**")
        st.success(f"**Risk Probability: {probability:.1f}%**")

# Streamlit App
def main():
    st.set_page_config(page_title="Diabetes Prediction", page_icon="💉")
    st.title("🩺 Diabetes Risk Assessment Tool")

    model, scaler, medians = load_resources()
    if model is None:
        st.stop()

    st.sidebar.header("📋 Patient Information")
    name = st.sidebar.text_input("👤 Full Name")

    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.number_input("Glucose (mg/dL)", 0, 300, 120)
    bloodpressure = st.sidebar.number_input("Blood Pressure (mmHg)", 0, 200, 80)
    skinthickness = st.sidebar.number_input("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.sidebar.number_input("Insulin (mu U/ml)", 0, 500, 0)
    bmi = st.sidebar.number_input("BMI", 0.0, 50.0, 25.0, format="%.1f")
    diabetespedigree = st.sidebar.number_input("Diabetes Pedigree", 0.0, 2.5, 0.5, format="%.3f")
    age = st.sidebar.number_input("Age", 1, 120, 30)

    if st.sidebar.button("🔍 Assess Risk"):
        errors, warnings = validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age)
        if errors:
            for e in errors:
                st.error(f"• {e}")
            return
        prediction, probability = predict_diabetes(model, scaler, medians, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age)
        display_results(name, prediction, probability, errors, warnings)

if __name__ == "__main__":
    main()
