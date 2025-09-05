"""
Streamlit diabetes risk app
- Fetches model artifacts from models/
- Predicts diabetes risk
- Logs each prediction by fetching the current audit_log.csv from HF,
  appending the new row, and re-uploading the merged CSV.
This avoids overwrites and preserves history across restarts.
"""

import os
import time
import tempfile
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from huggingface_hub import HfApi, hf_hub_download
# `create_repo` kept for initial repo creation if needed:
from huggingface_hub import create_repo

# -----------------------
# Basic config
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
# model artifact filenames (adjust if your filenames differ)
MODEL_FILE = MODEL_DIR / "diabetes.sav"
SCALER_FILE = MODEL_DIR / "scaler.sav"
MEDIANS_FILE = MODEL_DIR / "medians.sav"

HF_USERNAME = "LovnishVerma"
DATASET_REPO = f"{HF_USERNAME}/diabetes-logs"  # repo id
HF_TOKEN = os.getenv("HF_TOKEN")  # must be a write-capable token

# -----------------------
# Helpers for Hugging Face dataset
# -----------------------
def ensure_dataset_repo():
    """
    Create dataset repo (no-op if exists). Requires HF_TOKEN with write permissions.
    Runs once per session.
    """
    try:
        create_repo(
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            private=False,
            repo_type="dataset",
            exist_ok=True,
        )
        # Optionally push a README if it doesn't exist (silent if it does)
        api = HfApi()
        api.upload_file(
            path_or_fileobj="# Diabetes Risk Assessment Logs\nAuto-updated by Streamlit app.".encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
    except Exception as e:
        # Do not crash here — just log. Repo may already exist or token may be limited.
        logger.info(f"ensure_dataset_repo: {e}")


def fetch_remote_logs_via_api(retries: int = 1, delay: float = 0.5) -> pd.DataFrame:
    """
    Robustly fetch the remote audit_log.csv from the HF dataset.
    Preferred method: hf_hub_download (works even when raw URL is restricted).
    Falls back to trying the raw URL via pandas if hf_hub_download errors.
    Returns an empty DataFrame when no remote file exists or fetch fails.
    """
    # 1) Try hf_hub_download (most reliable)
    if HF_TOKEN:
        for attempt in range(retries):
            try:
                local = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename="audit_log.csv",
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                df = pd.read_csv(local, dtype=str)
                logger.info("Fetched remote logs using hf_hub_download.")
                return df
            except Exception as e:
                logger.debug(f"hf_hub_download attempt {attempt+1} failed: {e}")
                time.sleep(delay)
    # 2) Fallback: try the public raw URL (works if dataset file is public)
    try:
        raw_url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
        df = pd.read_csv(raw_url, dtype=str)
        logger.info("Fetched remote logs via raw URL.")
        return df
    except Exception as e:
        logger.info(f"No remote logs available or fetch failed: {e}")

    # Return empty DataFrame if nothing succeeded
    return pd.DataFrame()


def upload_merged_logs(tmp_csv_path: str):
    """
    Upload the merged CSV file at tmp_csv_path back to the HF dataset.
    This overwrites the file in the dataset with the provided CSV (which should be the merged history).
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set — cannot upload logs to Hugging Face.")

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(tmp_csv_path),
            path_in_repo="audit_log.csv",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Update audit_log {datetime.utcnow().isoformat()}",
            create_pr=False,
        )
        logger.info("Uploaded merged audit_log.csv to Hugging Face.")
    except Exception as e:
        # Surface helpful message when token lacks permission (401/403)
        logger.exception("upload_merged_logs failed")
        raise


# -----------------------
# Model utilities
# -----------------------
@st.cache_resource
def load_resources():
    """
    Load model, scaler and medians from disk.
    """
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        medians = joblib.load(MEDIANS_FILE)

        # Basic sanity check for medians keys
        required = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"}
        if not required.issubset(set(medians.keys())):
            raise ValueError("medians object missing required keys")

        logger.info("Loaded model, scaler and medians.")
        return model, scaler, medians
    except Exception as e:
        logger.exception("load_resources failed")
        return None, None, None


# -----------------------
# Prediction code
# -----------------------
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors = []
    if not (0 < glucose <= 300): errors.append("Glucose must be 1–300 mg/dL.")
    if not (0 < bloodpressure <= 200): errors.append("Blood pressure must be 1–200 mmHg.")
    if not (0 < bmi <= 70): errors.append("BMI must be 1–70.")
    if not (0 < age <= 120): errors.append("Age must be 1–120.")
    if pregnancies > 20: errors.append("Pregnancies cannot exceed 20.")
    if age < 15 and pregnancies > 0: errors.append("Age too low for pregnancies.")
    return errors


def predict_diabetes(model, scaler, medians, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
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
        logger.exception("predict_diabetes failed")
        return None, None


# -----------------------
# Central logging operation (fetch -> append -> upload)
# -----------------------
def log_prediction_remote_only(name: str, inputs: dict, prediction: bool, probability: float):
    """
    Remote-first logging:
      1) fetch latest remote audit_log.csv (if exists)
      2) append the new row
      3) write merged CSV to a temp file
      4) upload merged CSV back to HF dataset
    This reduces the chance to overwrite history.
    """
    # 1) Fetch current remote logs (if any)
    logs = fetch_remote_logs_via_api(retries=2, delay=0.3)

    # 2) Build new row
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name or "Anonymous",
        "pregnancies": inputs.get("pregnancies"),
        "glucose": inputs.get("glucose"),
        "bloodpressure": inputs.get("bloodpressure"),
        "skinthickness": inputs.get("skinthickness"),
        "insulin": inputs.get("insulin"),
        "bmi": inputs.get("bmi"),
        "diabetespedigree": inputs.get("diabetespedigree"),
        "age": inputs.get("age"),
        "prediction": "Positive" if prediction else "Negative",
        "probability": f"{probability:.1f}%",
        "region": "India",
    }

    # 3) Append
    try:
        if logs is not None and not logs.empty:
            merged = pd.concat([logs, pd.DataFrame([new_row])], ignore_index=True)
        else:
            merged = pd.DataFrame([new_row])
    except Exception:
        # Defensive fallback
        merged = pd.DataFrame([new_row])

    # 4) Save merged to a temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmpf:
        tmp_path = Path(tmpf.name)
        merged.to_csv(tmp_path, index=False)

    # 5) Upload the merged file
    if not HF_TOKEN:
        # If token not present, give clear UI message but still keep app functional
        logger.warning("HF_TOKEN not set — not uploading logs to Hugging Face. Prediction logged only in memory for this session.")
        return

    try:
        upload_merged_logs(tmp_path)
    except Exception as e:
        # Surface a clear error for token/permission issues
        # do not delete tmp file so operator can inspect if needed
        logger.exception("Failed to upload merged logs to Hugging Face")
        st.error("Failed to upload logs to Hugging Face. Check HF_TOKEN permissions (needs write).")
        return
    finally:
        # Clean up tmp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="Diabetes Risk", page_icon="💉", layout="centered")
    st.title("🩺 Diabetes Risk Assessment")

    # Make sure dataset exists (best-effort)
    if "repo_setup" not in st.session_state:
        ensure_dataset_repo()
        st.session_state.repo_setup = True

    # Load model resources
    model, scaler, medians = load_resources()
    if model is None:
        st.stop()

    # Sidebar inputs
    with st.sidebar:
        st.header("Patient info")
        name = st.text_input("Name (optional)")
        pregnancies = st.number_input("Pregnancies", 0, 20, value=0)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, value=120)
        bloodpressure = st.number_input("Blood Pressure (mmHg)", 0, 200, value=80)
        skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, value=20)
        insulin = st.number_input("Insulin (μU/mL)", 0, 500, value=0)
        bmi = st.number_input("BMI", 0.0, 70.0, value=25.0, format="%.1f")
        diabetespedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, value=0.5, format="%.3f")
        age = st.number_input("Age", 1, 120, value=30)

        if st.button("Assess Risk"):
            inputs = {
                "pregnancies": pregnancies, "glucose": glucose, "bloodpressure": bloodpressure,
                "skinthickness": skinthickness, "insulin": insulin, "bmi": bmi,
                "diabetespedigree": diabetespedigree, "age": age
            }
            errors = validate_inputs(**inputs)
            if errors:
                st.error("Please fix input errors:")
                for e in errors:
                    st.write(f"- {e}")
            else:
                pred, prob = predict_diabetes(model, scaler, medians, **inputs)
                if pred is None:
                    st.error("Prediction failed (see logs).")
                else:
                    # Log remotely (fetch->append->upload)
                    log_prediction_remote_only(name, inputs, pred, prob)
                    label = "High risk" if pred else "Low risk"
                    if pred:
                        st.error(f"🔴 {label} — Risk: {prob:.1f}%")
                    else:
                        st.success(f"✅ {label} — Risk: {prob:.1f}%")

    # Show recent logs (always fetch remote latest)
    st.markdown("---")
    st.subheader("Recent predictions (last 5)")

    try:
        logs = fetch_remote_logs_via_api(retries=2, delay=0.3)
        if logs is not None and not logs.empty:
            # ensure timestamp is present and sort newest first
            if "timestamp" in logs.columns:
                logs_sorted = logs.sort_values("timestamp", ascending=False).head(5)
            else:
                logs_sorted = logs.tail(5)
            st.dataframe(logs_sorted)
        else:
            st.info("No logs available yet.")
    except Exception as e:
        logger.exception("Failed to load logs for display")
        st.info("Could not fetch logs right now. Try again later.")

if __name__ == "__main__":
    main()
