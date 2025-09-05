import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# --- Your Hugging Face Dataset (PUBLIC) ---
HF_USERNAME = "LovnishVerma"
DATASET_REPO = f"{HF_USERNAME}/diabetes-logs"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set in HF Secrets

# Create dataset repo if not exists (run once)
def ensure_dataset_repo():
    try:
        create_repo(DATASET_REPO, token=HF_TOKEN, private=False, repo_type="dataset", exist_ok=True)
        # Create README if needed
        api = HfApi()
        api.upload_file(
            path_or_fileobj="# Diabetes Risk Assessment Logs\nAuto-updated by Streamlit app.".encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset"
        )
    except Exception as e:
        logger.info(f"Dataset repo setup: {e}")

# Run once
if "repo_setup" not in st.session_state:
    ensure_dataset_repo()
    st.session_state.repo_setup = True

# --- Load Model, Scaler, Medians ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_DIR / "diabetes.sav")
        scaler = joblib.load(MODEL_DIR / "scaler.sav")
        medians = joblib.load(MODEL_DIR / "medians.sav")

        expected = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"}
        if not expected.issubset(medians.keys()):
            raise ValueError("Medians missing keys")

        logger.info("✅ Model assets loaded.")
        return model, scaler, medians
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        st.error("Model not found. Check models/")
        return None, None, None

# --- Input Validation ---
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors = []
    if not (0 < glucose <= 300): errors.append("Glucose: 1–300 mg/dL.")
    if not (0 < bloodpressure <= 200): errors.append("BP: 1–200 mmHg.")
    if not (0 < bmi <= 70): errors.append("BMI: 1–70.")
    if not (0 < age <= 120): errors.append("Age: 1–120 years.")
    if pregnancies > 20: errors.append("Pregnancies ≤ 20.")
    if age < 15 and pregnancies > 0: errors.append("Age too low for pregnancies.")
    return errors, []

# --- Predict ---
def predict_diabetes(model, scaler, medians, pregnancies, glucose, bloodpressure,
                     skinthickness, insulin, bmi, diabetespedigree, age):
    try:
        df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bloodpressure,
            "SkinThickness": skinthickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetespedigree,
            "Age": age
        }])

        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        df = df.fillna(medians)
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        return bool(pred), float(prob)
    except Exception as e:
        logger.error(f"Predict error: {e}")
        st.error("Prediction failed.")
        return None, None

# --- Log to Hugging Face Dataset (Public) ---
def log_prediction(name, inputs, prediction, probability):
    try:
        # New log entry
        new_log = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name or "Anonymous",
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
        }])

        # Load existing logs or start fresh
        try:
            # ✅ CORRECTED: No extra spaces in URL
            url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/audit_log.csv"
            existing = pd.read_csv(url)
            updated = pd.concat([existing, new_log], ignore_index=True)
        except Exception as e:
            logger.warning(f"Could not load existing logs: {e}")
            updated = new_log  # First entry

        # Save locally (for speed)
        updated.to_csv(LOG_DIR / "audit_log.csv", index=False)

        # Push to HF
        api = HfApi()
        api.upload_file(
            path_or_fileobj=LOG_DIR / "audit_log.csv",
            path_in_repo="audit_log.csv",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset"
        )
        logger.info("✅ Logged to HF Dataset")
    except Exception as e:
        logger.error(f"Log failed: {e}")

# --- Main App ---
def main():
    st.set_page_config(page_title="🩺 Diabetes Risk", page_icon="💉", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>AI Screening Tool • Powered by Hugging Face</p>", unsafe_allow_html=True)
    st.markdown("---")

    model, scaler, medians = load_resources()
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("📋 Patient Info")
        name = st.text_input("Name (Optional)", placeholder="e.g., Rajesh")

        st.markdown("### Clinical Inputs")
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        bloodpressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 80)
        skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        insulin = st.number_input("Insulin (μU/mL)", 0, 500, 0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, format="%.1f")
        diabetespedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, format="%.3f")
        age = st.number_input("Age", 1, 120, 30)

        # Track input changes
        current_inputs = {
            "name": name, "pregnancies": pregnancies, "glucose": glucose,
            "bloodpressure": bloodpressure, "skinthickness": skinthickness,
            "insulin": insulin, "bmi": bmi, "diabetespedigree": diabetespedigree, "age": age
        }

        if st.session_state.get("last_inputs") != current_inputs:
            st.session_state.pop("run_prediction", None)
        st.session_state.last_inputs = current_inputs

        st.markdown("---")
        if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
            st.session_state.run_prediction = True
            st.session_state.inputs = current_inputs.copy()

    # Run Prediction
    if st.session_state.get("run_prediction") and "inputs" in st.session_state:
        inputs = st.session_state.inputs.copy()
        name = inputs.pop("name", "Patient")

        errors, _ = validate_inputs(**inputs)
        if errors:
            st.error("🔴 **Errors**")
            for e in errors:
                st.write(f"• {e}")
            st.session_state.run_prediction = False
            return

        with st.spinner("Analyzing..."):
            pred, prob = predict_diabetes(model, scaler, medians, **inputs)

        if pred is None:
            return

        # Log to HF
        log_prediction(name, inputs, pred, prob)

        # Show Result
        st.markdown("---")
        if pred:
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background:#ffebee; border-left:5px solid #f44336; color:#c62828;">
                <h3>🔴 High Risk</h3>
                <p><strong>{name}</strong>, AI detected <strong>high risk</strong>.</p>
                <p><strong>Risk: {prob:.1f}%</strong></p>
                <p><em>Consult a doctor for HbA1c test.</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background:#e8f5e8; border-left:5px solid #4caf50; color:#2e7d32;">
                <h3>✅ Low Risk</h3>
                <p><strong>{name}</strong>, your risk is currently <strong>low</strong>.</p>
                <p><strong>Risk: {prob:.1f}%</strong></p>
                <p><em>Maintain healthy lifestyle.</em></p>
            </div>
            """, unsafe_allow_html=True)

        # Download report
        report = f"Diabetes Risk Report\nPatient: {name}\nRisk: {'High' if pred else 'Low'}\nProbability: {prob:.1f}%\nDate: {datetime.now()}"
        st.download_button("📥 Download Report", report, "report.txt")

        st.session_state.run_prediction = False

    # Optional: Show public logs
    if st.checkbox("📊 View Public Logs (last 10)"):
        try:
            # ✅ CORRECTED: No extra spaces
            url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/audit_log.csv"
            logs = pd.read_csv(url)
            st.dataframe(logs.tail(10))
        except Exception as e:
            st.info("No logs yet. Make a prediction!")

if __name__ == "__main__":
    main()