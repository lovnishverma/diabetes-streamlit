import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os

# ===============================
# 🔧 Setup
# ===============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")

HF_USERNAME = "LovnishVerma"
DATASET_REPO = f"{HF_USERNAME}/diabetes-logs"
HF_TOKEN = os.getenv("HF_TOKEN")  # make sure this is set in Streamlit Secrets

# ===============================
# 🚀 Hugging Face Repo
# ===============================
def ensure_dataset_repo():
    try:
        create_repo(
            DATASET_REPO,
            token=HF_TOKEN,
            private=False,
            repo_type="dataset",
            exist_ok=True,
        )
        api = HfApi()
        api.upload_file(
            path_or_fileobj="# Diabetes Risk Assessment Logs\nAuto-updated by Streamlit app.".encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
    except Exception as e:
        logger.warning(f"Repo setup issue: {e}")

if "repo_setup" not in st.session_state:
    ensure_dataset_repo()
    st.session_state.repo_setup = True

# ===============================
# 📦 Load Model + Scaler + Medians
# ===============================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_DIR / "diabetes.sav")
        scaler = joblib.load(MODEL_DIR / "scaler.sav")
        medians = joblib.load(MODEL_DIR / "medians.sav")
        return model, scaler, medians
    except Exception as e:
        st.error("Model files not found. Please upload them in 'models/'")
        logger.error(e)
        return None, None, None

# ===============================
# 🩺 Validation
# ===============================
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness,
                    insulin, bmi, diabetespedigree, age):
    errors = []
    if not (0 < glucose <= 300): errors.append("Glucose must be 1–300.")
    if not (0 < bloodpressure <= 200): errors.append("Blood Pressure 1–200.")
    if not (0 < bmi <= 70): errors.append("BMI must be 1–70.")
    if not (0 < age <= 120): errors.append("Age must be 1–120.")
    if pregnancies > 20: errors.append("Pregnancies cannot exceed 20.")
    if age < 15 and pregnancies > 0: errors.append("Age too low for pregnancies.")
    return errors

# ===============================
# 🤖 Predict
# ===============================
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
        # replace zeros with medians
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        df = df.fillna(medians)

        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        return bool(pred), float(prob)
    except Exception as e:
        logger.error(e)
        st.error("Prediction failed.")
        return None, None

# ===============================
# 📊 Logging
# ===============================
def fetch_remote_logs():
    try:
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
        return pd.read_csv(url, dtype=str)
    except Exception as e:
        logger.info(f"No logs yet: {e}")
        return pd.DataFrame()

def log_prediction(name, inputs, prediction, probability):
    try:
        # Fetch old logs
        logs = fetch_remote_logs()

        # New row
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

        # Merge with existing
        updated = pd.concat([logs, new_log], ignore_index=True)

        # Save temp CSV
        tmp_file = Path("audit_log.csv")
        updated.to_csv(tmp_file, index=False)

        # Upload back
        if HF_TOKEN:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(tmp_file),
                path_in_repo="audit_log.csv",
                repo_id=DATASET_REPO,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Update logs {datetime.now().isoformat()}",
                create_pr=False,
            )
        else:
            st.warning("HF_TOKEN not set → logs not uploaded")
    except Exception as e:
        logger.error(f"Log update failed: {e}")

# ===============================
# 🎨 Streamlit App
# ===============================
def main():
    st.set_page_config(page_title="🩺 Diabetes Risk", page_icon="💉")

    st.title("🩺 Diabetes Risk Assessment")

    model, scaler, medians = load_resources()
    if model is None: st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Patient Info")
        name = st.text_input("Name (Optional)")
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        bloodpressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 80)
        skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 500, 0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, format="%.1f")
        diabetespedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, format="%.3f")
        age = st.number_input("Age", 1, 120, 30)

        inputs = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetespedigree": diabetespedigree,
            "age": age,
        }

        if st.button("🔍 Assess Risk"):
            errors = validate_inputs(**inputs)
            if errors:
                st.error("Please fix:")
                for e in errors: st.write(f"- {e}")
            else:
                pred, prob = predict_diabetes(model, scaler, medians, **inputs)
                if pred is not None:
                    log_prediction(name, inputs, pred, prob)
                    st.success(f"Risk: {'Positive' if pred else 'Negative'} ({prob:.1f}%)")

    # Show recent logs
    st.subheader("📊 Recent Predictions")
    logs = fetch_remote_logs()
    if not logs.empty:
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(5))
    else:
        st.info("No logs available yet.")

if __name__ == "__main__":
    main()
