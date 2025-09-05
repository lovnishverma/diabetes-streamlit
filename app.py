import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# --- Load Model, Scaler, Medians ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_DIR / "diabetes.sav")
        scaler = joblib.load(MODEL_DIR / "scaler.sav")
        medians = joblib.load(MODEL_DIR / "medians.sav")  # dict

        # Validate required keys
        expected_features = {
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        }
        if not expected_features.issubset(medians.keys()):
            missing = expected_features - set(medians.keys())
            raise ValueError(f"Medians missing keys: {missing}")

        logger.info("✅ Model, scaler, and medians loaded.")
        return model, scaler, medians

    except Exception as e:
        logger.error(f"❌ Failed to load resources: {e}")
        st.error("🚨 System Error: Could not load AI model. Contact administrator.")
        return None, None, None


# --- Input Validation ---
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors = []
    warnings = []

    if not (0 < glucose <= 300):
        errors.append("Glucose must be 1–300 mg/dL.")
    if not (0 < bloodpressure <= 200):
        errors.append("Blood Pressure must be 1–200 mmHg.")
    if not (0 < bmi <= 70):
        errors.append("BMI must be 1–70 kg/m².")
    if not (0 < age <= 120):
        errors.append("Age must be 1–120 years.")
    if pregnancies > 20:
        errors.append("Pregnancies should not exceed 20.")
    if age < 15 and pregnancies > 0:
        errors.append("❌ Age too low for pregnancy count.")
    if insulin > 300 and glucose < 90:
        warnings.append("⚠️ High insulin with low glucose – possible hypoglycemia.")

    return errors, warnings


# --- Predict Diabetes ---
def predict_diabetes(model, scaler, medians, pregnancies, glucose, bloodpressure,
                     skinthickness, insulin, bmi, diabetespedigree, age):
    try:
        # Use exact column names from training
        input_dict = {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [bloodpressure],
            "SkinThickness": [skinthickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetespedigree],
            "Age": [age]
        }
        df = pd.DataFrame(input_dict)

        # Replace 0s with NaN
        zero_sensitive = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_sensitive] = df[zero_sensitive].replace(0, np.nan)

        # Fill with training medians
        df = df.fillna(value=medians)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1] * 100

        return bool(prediction), float(probability)

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        st.error("🔧 Internal error during prediction.")
        return None, None


# --- Audit Logging ---
def log_prediction(name, inputs, prediction, probability):
    log_file = LOG_DIR / "audit_log.csv"
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": name,
        "pregnancies": inputs["pregnancies"],
        "glucose": inputs["glucose"],
        "bloodpressure": inputs["bloodpressure"],
        "skinthickness": inputs["skinthickness"],
        "insulin": inputs["insulin"],
        "bmi": inputs["bmi"],
        "diabetespedigree": inputs["diabetespedigree"],
        "age": inputs["age"],
        "prediction": "Positive" if prediction else "Negative",
        "probability": f"{probability:.1f}%",
        "region": "India"
    }
    pd.DataFrame([entry]).to_csv(log_file, mode='a', header=not log_file.exists(), index=False)


# --- Explain Prediction ---
def show_explanation():
    st.markdown("### 🔍 How This Works")
    st.markdown("""
    - **AI Model**: Random Forest trained on clinical data.
    - **Key Factors**: Glucose, BMI, family history, and age.
    - **Purpose**: Early screening tool — not a diagnosis.
    - *Complies with India’s Digital Personal Data Protection (DPDPA) Act.*
    """)


# --- Main App ---
def main():
    st.set_page_config(
        page_title="🏥 MedCare AI: Diabetes Risk",
        page_icon="🩺",
        layout="centered"
    )

    st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>AI-powered screening • Trusted in Indian Hospitals</p>", unsafe_allow_html=True)
    st.markdown("---")

    model, scaler, medians = load_resources()
    if model is None:
        st.stop()

    # Sidebar Inputs
    with st.sidebar:
        st.header("📋 Patient Information")
        name = st.text_input("👤 Full Name", placeholder="e.g., Priya Sharma")

        st.markdown("### Clinical Parameters")
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Fasting Glucose (mg/dL)", 0, 300, 120)
        bloodpressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 80)
        skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        insulin = st.number_input("Insulin (μU/mL)", 0, 500, 0)
        bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, format="%.1f")
        diabetespedigree = st.number_input(
            "Diabetes Pedigree Function",
            0.0, 3.0, 0.5,
            format="%.3f",
            help="Genetic risk score based on family history"
        )
        age = st.number_input("Age (years)", 1, 120, 30)

        # Create a unique key for current input state
        current_inputs = {
            "name": name,
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetespedigree": diabetespedigree,
            "age": age
        }

        # Reset prediction if inputs changed
        if st.session_state.get("last_inputs") != current_inputs:
            st.session_state.pop("run_prediction", None)

        st.session_state.last_inputs = current_inputs

        st.markdown("---")
        if st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True):
            st.session_state.run_prediction = True
            st.session_state.inputs = current_inputs.copy()

    # Run Prediction Only If Requested
    if st.session_state.get("run_prediction") and "inputs" in st.session_state:
        inputs = st.session_state.inputs.copy()
        name = inputs.pop("name", "Patient")  # Safe pop with default

        errors, warnings = validate_inputs(**inputs)
        if errors:
            st.error("🔴 **Input Errors**")
            for e in errors:
                st.write(f"• {e}")
            st.session_state.run_prediction = False
            return

        if warnings:
            st.warning("⚠️ **Warnings**")
            for w in warnings:
                st.write(f"• {w}")

        with st.spinner("Analyzing clinical data..."):
            prediction, probability = predict_diabetes(model, scaler, medians, **inputs)

        if prediction is None:
            st.error("🔴 Prediction failed. Please try again.")
            st.session_state.run_prediction = False
            return

        # Log the result
        log_prediction(name, inputs, prediction, probability)

        # Display Result
        st.markdown("---")
        if prediction:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #ffebee; border-left: 5px solid #f44336; color: #c62828;">
                <h3>🔴 High Risk of Diabetes</h3>
                <p><strong>Hello {name},</strong> you are at <strong>high risk</strong>.</p>
                <p style="font-size: 1.2em;">🩸 Risk Probability: <strong>{probability:.1f}%</strong></p>
                <p><em>🩺 Recommendation: Consult a physician and perform an HbA1c or OGTT test.</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #e8f5e8; border-left: 5px solid #4caf50; color: #2e7d32;">
                <h3>✅ Low Risk of Diabetes</h3>
                <p><strong>Hello {name},</strong> your risk is currently <strong>low</strong>.</p>
                <p style="font-size: 1.2em;">📊 Risk Score: <strong>{probability:.1f}%</strong></p>
                <p><em>💡 Tip: Maintain healthy diet and regular exercise to stay protected.</em></p>
            </div>
            """, unsafe_allow_html=True)

        # Show explanation
        show_explanation()

        # Generate Report
        report = f"""
        Diabetes Risk Assessment Report
        ===============================
        Patient: {name}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Risk Level: {'High' if prediction else 'Low'}
        Probability: {probability:.1f}%
        
        Clinical Inputs:
          - Glucose: {inputs['glucose']} mg/dL
          - BMI: {inputs['bmi']} kg/m²
          - Age: {inputs['age']} years
          - Family Risk (Pedigree): {inputs['diabetespedigree']:.3f}
        
        Disclaimer: This is a screening tool, not a diagnosis.
        """
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"Diabetes_Risk_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

        # Prevent auto-rerun
        st.session_state.run_prediction = False


if __name__ == "__main__":
    main()