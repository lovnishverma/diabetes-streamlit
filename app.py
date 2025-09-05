import os
import time
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from huggingface_hub import HfApi, hf_hub_download, create_repo

# -----------------------
# Logging setup (rotating file logs + console)
# -----------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

handler = RotatingFileHandler(LOG_FILE, maxBytes=500_000, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[handler, logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "diabetes.sav"
SCALER_FILE = MODEL_DIR / "scaler.sav"
MEDIANS_FILE = MODEL_DIR / "medians.sav"

HF_USERNAME = "LovnishVerma"
DATASET_REPO = f"{HF_USERNAME}/diabetes-logs"
HF_TOKEN = os.getenv("HF_TOKEN")  # write-capable token required


# -----------------------
# Hugging Face helpers
# -----------------------
def ensure_dataset_repo():
    """Create dataset repo if not exists."""
    try:
        create_repo(
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            private=False,
            repo_type="dataset",
            exist_ok=True,
        )
        api = HfApi()
        api.upload_file(
            path_or_fileobj="# Diabetes Risk Logs\nAuto-updated by Streamlit.".encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
    except Exception as e:
        logger.info(f"Repo setup skipped: {e}")


def fetch_remote_logs(retries: int = 1, delay: float = 0.5) -> pd.DataFrame:
    """Fetch audit_log.csv from HF (dataset repo)."""
    if HF_TOKEN:
        for attempt in range(retries):
            try:
                local = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename="audit_log.csv",
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                return pd.read_csv(local, dtype=str)
            except Exception as e:
                logger.debug(f"Download attempt {attempt+1} failed: {e}")
                time.sleep(delay)

    try:
        raw_url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
        return pd.read_csv(raw_url, dtype=str)
    except Exception as e:
        logger.info(f"No remote logs found: {e}")
        return pd.DataFrame()


def upload_merged_logs(tmp_csv: str):
    """Upload updated logs to HF."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set — cannot upload logs.")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=tmp_csv,
        path_in_repo="audit_log.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Update audit_log {datetime.utcnow().isoformat()}",
        create_pr=False,
    )
    logger.info("Uploaded audit_log.csv")


# -----------------------
# Model utils
# -----------------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        medians = joblib.load(MEDIANS_FILE)
        return model, scaler, medians
    except Exception as e:
        logger.exception("Model load failed")
        return None, None, None


def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors = []
    if not (1 <= glucose <= 300): errors.append("Glucose must be 1–300 mg/dL.")
    if not (1 <= bloodpressure <= 200): errors.append("Blood pressure must be 1–200 mmHg.")
    if not (1 <= bmi <= 70): errors.append("BMI must be 1–70.")
    if not (1 <= age <= 120): errors.append("Age must be 1–120.")
    if pregnancies > 20: errors.append("Pregnancies cannot exceed 20.")
    if age < 15 and pregnancies > 0: errors.append("Age too low for pregnancies.")
    return errors


def predict(model, scaler, medians, **inputs):
    try:
        df = pd.DataFrame([inputs])
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_cols] = df[zero_cols].replace(0, np.nan).fillna(medians)
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        return bool(pred), float(prob)
    except Exception:
        logger.exception("Prediction failed")
        return None, None


def log_prediction(name, inputs, prediction, probability):
    logs = fetch_remote_logs()
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name or "Anonymous",
        **inputs,
        "prediction": "Positive" if prediction else "Negative",
        "probability": f"{probability:.1f}%",
        "region": "India",
    }
    merged = pd.concat([logs, pd.DataFrame([new_row])], ignore_index=True) if not logs.empty else pd.DataFrame([new_row])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmpf:
        tmp_path = Path(tmpf.name)
        merged.to_csv(tmp_path, index=False)

    if HF_TOKEN:
        try:
            upload_merged_logs(tmp_path)
        except Exception as e:
            logger.error("Upload failed")
            st.error("⚠️ Could not upload logs (check HF_TOKEN permissions).")


# -----------------------
# UI
# -----------------------
def main():
    st.set_page_config(page_title="Diabetes Risk", page_icon="💉", layout="wide")
    st.title("🩺 Diabetes Risk Assessment")

    if "repo_setup" not in st.session_state:
        ensure_dataset_repo()
        st.session_state.repo_setup = True

    model, scaler, medians = load_resources()
    if model is None:
        st.error("Model not available. Please check deployment.")
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Information")
        with st.expander("Fill details"):
            name = st.text_input("Name (optional)")
            pregnancies = st.number_input("Pregnancies", 0, 20, value=0)
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, value=120)
            bloodpressure = st.number_input("Blood Pressure (mmHg)", 0, 200, value=80)
            skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, value=20)
            insulin = st.number_input("Insulin (μU/mL)", 0, 500, value=0)
            bmi = st.number_input("BMI", 0.0, 70.0, value=25.0, format="%.1f")
            diabetespedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, value=0.5, format="%.3f")
            age = st.number_input("Age", 1, 120, value=30)

        if st.button("⚡ Assess Risk", use_container_width=True):
            inputs = dict(
                pregnancies=pregnancies, glucose=glucose, bloodpressure=bloodpressure,
                skinthickness=skinthickness, insulin=insulin, bmi=bmi,
                diabetespedigree=diabetespedigree, age=age
            )
            errors = validate_inputs(**inputs)
            if errors:
                st.error("❌ Invalid inputs:")
                for e in errors: st.write(f"- {e}")
            else:
                pred, prob = predict(model, scaler, medians, **inputs)
                if pred is None:
                    st.error("Prediction failed.")
                else:
                    log_prediction(name, inputs, pred, prob)
                    if pred:
                        st.error(f"🔴 High Risk — {prob:.1f}%")
                    else:
                        st.success(f"✅ Low Risk — {prob:.1f}%")
                    st.progress(min(int(prob), 100))

    with col2:
        st.subheader("📊 Recent Predictions")
        logs = fetch_remote_logs()
        if not logs.empty:
            logs = logs.sort_values("timestamp", ascending=False).head(8)
            st.dataframe(logs, use_container_width=True, height=300)
        else:
            st.info("No logs yet.")


if __name__ == "__main__":
    main()
