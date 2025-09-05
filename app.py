import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os

# 🔧 Basic setup: logging, folders, Hugging Face config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "audit_log.csv"

HF_USERNAME = "LovnishVerma"
DATASET_REPO = f"{HF_USERNAME}/diabetes-logs"
HF_TOKEN = os.getenv("HF_TOKEN")  # should be set in Streamlit secrets/environment

# Ensure Hugging Face dataset repo exists
def ensure_dataset_repo():
    try:
        create_repo(
            DATASET_REPO,
            token=HF_TOKEN,
            private=False,
            repo_type="dataset",
            exist_ok=True,
        )
        # Push a README if missing
        api = HfApi()
        api.upload_file(
            path_or_fileobj="# Diabetes Risk Assessment Logs\nAuto-updated by Streamlit app.".encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
    except Exception as e:
        logger.info(f"Dataset repo setup issue: {e}")

if "repo_setup" not in st.session_state:
    ensure_dataset_repo()
    st.session_state.repo_setup = True

# 🔄 Sync logs from Hugging Face to local on startup
def sync_logs_on_start():
    """Ensure local logs are synced with Hugging Face dataset on startup."""
    try:
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
        remote_logs = pd.read_csv(url, dtype=str)
        remote_logs.to_csv(LOG_FILE, index=False)  # overwrite local with remote
        logger.info("Synced logs from Hugging Face into local file.")
    except Exception as e:
        logger.warning(f"Could not sync logs on startup: {e}")

# 📦 Load model, scaler and medians
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_DIR / "diabetes.sav")
        scaler = joblib.load(MODEL_DIR / "scaler.sav")
        medians = joblib.load(MODEL_DIR / "medians.sav")

        expected = {
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age",
        }
        if not expected.issubset(medians.keys()):
            raise ValueError("Medians file missing required columns")

        logger.info("Model, scaler and medians loaded successfully.")
        return model, scaler, medians
    except Exception as e:
        logger.error(f"Resource load failed: {e}")
        st.error("Model files not found. Please check 'models/' directory.")
        return None, None, None

# ✅ Input validation
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness,
                    insulin, bmi, diabetespedigree, age):
    errors = []
    if not (0 < glucose <= 300):
        errors.append("Glucose must be between 1–300 mg/dL.")
    if not (0 < bloodpressure <= 200):
        errors.append("Blood Pressure must be between 1–200 mmHg.")
    if not (0 < bmi <= 70):
        errors.append("BMI must be between 1–70.")
    if not (0 < age <= 120):
        errors.append("Age must be between 1–120 years.")
    if pregnancies > 20:
        errors.append("Pregnancies cannot exceed 20.")
    if age < 15 and pregnancies > 0:
        errors.append("Age too low for pregnancies.")
    return errors

# 🤖 Run prediction
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
            "Age": age,
        }])

        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        df = df.fillna(medians)

        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        return bool(pred), float(prob)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error("Prediction failed. Check inputs or model.")
        return None, None

# 📝 Log results locally + sync with Hugging Face
def log_prediction(name, inputs, prediction, probability):
    try:
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
            "region": "India",
        }])

        # --- Local log path ---
        local_path = LOG_DIR / "audit_log.csv"

        # Always load existing logs first (from local or remote if missing)
        if local_path.exists():
            existing = pd.read_csv(local_path)
            updated = pd.concat([existing, new_log], ignore_index=True)
        else:
            # If local file missing (e.g., after restart), pull from remote
            try:
                url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
                remote = pd.read_csv(url, dtype=str)
                updated = pd.concat([remote, new_log], ignore_index=True)
            except Exception:
                updated = new_log  # no remote available

        # Save locally
        updated.to_csv(local_path, index=False)

        # Push the whole updated file to Hugging Face
        if HF_TOKEN:
            try:
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo="audit_log.csv",
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    commit_message=f"Log update {datetime.now().isoformat()}",
                    create_pr=False,
                )
                logger.info("✅ Logs appended and synced with Hugging Face.")
            except Exception as e:
                logger.warning(f"Hugging Face upload failed: {e}")
        else:
            st.warning("⚠️ HF_TOKEN not set → logs saved locally only.")

    except Exception as e:
        logger.error(f"❌ Log save failed: {e}")
        st.error("Could not save logs.")


# 🎨 Main Streamlit App
def main():
    st.set_page_config(page_title="🩺 Diabetes Risk", page_icon="💉", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>AI Screening Tool • Powered by Hugging Face</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 🔄 Sync logs on startup
    sync_logs_on_start()

    model, scaler, medians = load_resources()
    if model is None:
        st.stop()

    # ---------------- Sidebar Inputs ----------------
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

        current_inputs = {
            "name": name,
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetespedigree": diabetespedigree,
            "age": age,
        }

        if st.session_state.get("last_inputs") != current_inputs:
            st.session_state.pop("run_prediction", None)
        st.session_state.last_inputs = current_inputs

        st.markdown("---")
        if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
            st.session_state.run_prediction = True
            st.session_state.inputs = current_inputs.copy()

    # ---------------- Prediction ----------------
    if st.session_state.get("run_prediction") and "inputs" in st.session_state:
        inputs = st.session_state.inputs.copy()
        patient_name = inputs.pop("name", "Patient")

        errors = validate_inputs(**inputs)
        if errors:
            st.error("⚠️ Please fix the following:")
            for e in errors:
                st.write(f"- {e}")
            st.session_state.run_prediction = False
            return

        with st.spinner("Analyzing..."):
            pred, prob = predict_diabetes(model, scaler, medians, **inputs)

        if pred is None:
            return

        log_prediction(patient_name, inputs, pred, prob)

        st.markdown("---")
        if pred:
            st.markdown(
                f"""
                <div style="padding:20px; border-radius:10px; background:#ffebee; border-left:5px solid #f44336; color:#c62828;">
                    <h3>🔴 High Risk</h3>
                    <p><strong>{patient_name}</strong>, AI detected a <strong>high diabetes risk</strong>.</p>
                    <p><strong>Risk Score: {prob:.1f}%</strong></p>
                    <p><em>Please consult a doctor for further tests (e.g. HbA1c).</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding:20px; border-radius:10px; background:#e8f5e8; border-left:5px solid #4caf50; color:#2e7d32;">
                    <h3>✅ Low Risk</h3>
                    <p><strong>{patient_name}</strong>, your current risk is <strong>low</strong>.</p>
                    <p><strong>Risk Score: {prob:.1f}%</strong></p>
                    <p><em>Keep maintaining a healthy lifestyle.</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        report = (
            f"Diabetes Risk Report\n"
            f"Patient: {patient_name}\n"
            f"Risk: {'High' if pred else 'Low'}\n"
            f"Probability: {prob:.1f}%\n"
            f"Date: {datetime.now()}\n"
        )
        st.download_button("📥 Download Report", report, "report.txt")
        st.session_state.run_prediction = False

    # ---------------- Recent Logs ----------------
    st.markdown("---")
    st.subheader("📊 Recent Predictions (Last 5)")
    try:
        logs = pd.read_csv(LOG_FILE, dtype=str)
        logs = logs.sort_values("timestamp", ascending=False).head(5)
        st.dataframe(logs)
    except Exception as e:
        st.info("⚠️ No logs available.")
        logger.warning(f"Log fetch failed: {e}")

# 🚀 Run app
if __name__ == "__main__":
    main()
