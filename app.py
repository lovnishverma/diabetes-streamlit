"""
Enhanced Streamlit Diabetes Risk App
- Clean UI with color-coded risk cards
- Sidebar input panel
- Interactive recent logs table
- Remote-first logging to Hugging Face
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
from huggingface_hub import HfApi, hf_hub_download, create_repo

# -----------------------
# Basic Config
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "diabetes.sav"
SCALER_FILE = MODEL_DIR / "scaler.sav"
MEDIANS_FILE = MODEL_DIR / "medians.sav"

HF_USERNAME = "LovnishVerma"
DATASET_REPO = f"{HF_USERNAME}/diabetes-logs"
HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------
# HF Dataset Helpers
# -----------------------
def ensure_dataset_repo():
    try:
        create_repo(DATASET_REPO, token=HF_TOKEN, private=False, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.upload_file(
            path_or_fileobj="# Diabetes Risk Assessment Logs\nAuto-updated by Streamlit app.".encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
        )
    except Exception as e:
        logger.info(f"ensure_dataset_repo: {e}")

def fetch_remote_logs_via_api(retries: int = 1, delay: float = 0.5) -> pd.DataFrame:
    if HF_TOKEN:
        for attempt in range(retries):
            try:
                local = hf_hub_download(repo_id=DATASET_REPO, filename="audit_log.csv", repo_type="dataset", token=HF_TOKEN)
                return pd.read_csv(local, dtype=str)
            except Exception as e:
                time.sleep(delay)
    try:
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
        return pd.read_csv(url, dtype=str)
    except Exception:
        return pd.DataFrame()

def upload_merged_logs(tmp_csv_path: str):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set — cannot upload logs to Hugging Face.")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(tmp_csv_path),
        path_in_repo="audit_log.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Update audit_log {datetime.utcnow().isoformat()}",
        create_pr=False,
    )

# -----------------------
# Model Utilities
# -----------------------
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        medians = joblib.load(MEDIANS_FILE)
        required = {"Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"}
        if not required.issubset(set(medians.keys())):
            raise ValueError("medians object missing required keys")
        return model, scaler, medians
    except Exception:
        return None, None, None

def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors=[]
    if not (0<glucose<=300): errors.append("Glucose must be 1–300 mg/dL.")
    if not (0<bloodpressure<=200): errors.append("Blood pressure must be 1–200 mmHg.")
    if not (0<bmi<=70): errors.append("BMI must be 1–70.")
    if not (0<age<=120): errors.append("Age must be 1–120.")
    if pregnancies>20: errors.append("Pregnancies cannot exceed 20.")
    if age<15 and pregnancies>0: errors.append("Age too low for pregnancies.")
    return errors

def predict_diabetes(model, scaler, medians, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    df=pd.DataFrame([{
        "Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bloodpressure,
        "SkinThickness":skinthickness,"Insulin":insulin,"BMI":bmi,
        "DiabetesPedigreeFunction":diabetespedigree,"Age":age
    }])
    zero_cols=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[zero_cols]=df[zero_cols].replace(0,np.nan)
    df=df.fillna(medians)
    scaled=scaler.transform(df)
    pred=model.predict(scaled)[0]
    prob=model.predict_proba(scaled)[0][1]*100
    return bool(pred), float(prob)

def log_prediction_remote_only(name: str, inputs: dict, prediction: bool, probability: float):
    logs=fetch_remote_logs_via_api(retries=2,delay=0.3)
    new_row={
        "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name":name or "Anonymous",
        "pregnancies":inputs.get("pregnancies"),
        "glucose":inputs.get("glucose"),
        "bloodpressure":inputs.get("bloodpressure"),
        "skinthickness":inputs.get("skinthickness"),
        "insulin":inputs.get("insulin"),
        "bmi":inputs.get("bmi"),
        "diabetespedigree":inputs.get("diabetespedigree"),
        "age":inputs.get("age"),
        "prediction":"Positive" if prediction else "Negative",
        "probability":f"{probability:.1f}%",
        "region":"India",
    }
    merged = pd.concat([logs,pd.DataFrame([new_row])],ignore_index=True) if not logs.empty else pd.DataFrame([new_row])
    with tempfile.NamedTemporaryFile(mode="w",suffix=".csv",delete=False) as tmpf:
        tmp_path=Path(tmpf.name)
        merged.to_csv(tmp_path,index=False)
    if HF_TOKEN:
        upload_merged_logs(tmp_path)
    try: tmp_path.unlink()
    except Exception: pass

# -----------------------
# UI
# -----------------------
def main():
    st.set_page_config(page_title="🩺 Diabetes Risk",page_icon="💉",layout="wide")
    st.markdown("<h1 style='text-align:center'>🩺 Diabetes Risk Assessment</h1>",unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;'>AI Screening Tool • Powered by Hugging Face</p>",unsafe_allow_html=True)
    st.markdown("---")

    if "repo_setup" not in st.session_state:
        ensure_dataset_repo()
        st.session_state.repo_setup=True

    model, scaler, medians=load_resources()
    if model is None:
        st.error("Model resources not found!")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Patient Info")
        name = st.text_input("Name (optional)")
        pregnancies = st.number_input("Pregnancies",0,20,value=0)
        glucose = st.number_input("Glucose (mg/dL)",0,300,value=120)
        bloodpressure = st.number_input("Blood Pressure (mmHg)",0,200,value=80)
        skinthickness = st.number_input("Skin Thickness (mm)",0,100,value=20)
        insulin = st.number_input("Insulin (μU/mL)",0,500,value=0)
        bmi = st.number_input("BMI",0.0,70.0,value=25.0,format="%.1f")
        diabetespedigree = st.number_input("Diabetes Pedigree",0.0,3.0,value=0.5,format="%.3f")
        age = st.number_input("Age",1,120,value=30)

        if st.button("Assess Risk"):
            inputs={
                "pregnancies":pregnancies,"glucose":glucose,"bloodpressure":bloodpressure,
                "skinthickness":skinthickness,"insulin":insulin,"bmi":bmi,
                "diabetespedigree":diabetespedigree,"age":age
            }
            errors=validate_inputs(**inputs)
            if errors:
                st.error("Please fix input errors:")
                for e in errors: st.write(f"- {e}")
            else:
                pred, prob=predict_diabetes(model, scaler, medians, **inputs)
                if pred is None:
                    st.error("Prediction failed (see logs).")
                else:
                    log_prediction_remote_only(name, inputs, pred, prob)
                    # --- Fancy risk card ---
                    col1,col2=st.columns([1,3])
                    with col1:
                        st.metric("Risk Score",f"{prob:.1f}%","🔴 High" if pred else "✅ Low")
                    with col2:
                        color="#ffcccc" if pred else "#ccffcc"
                        st.markdown(f"""
                        <div style="padding:15px;border-radius:10px;background:{color};">
                        <h3 style="margin:0">{'High Diabetes Risk' if pred else 'Low Diabetes Risk'}</h3>
                        <p>Patient: {name or 'Anonymous'}</p>
                        </div>
                        """,unsafe_allow_html=True)

    # Recent logs
    st.markdown("---")
    st.subheader("Recent Predictions")
    logs=fetch_remote_logs_via_api(retries=2,delay=0.3)
    if not logs.empty:
        logs_sorted=logs.sort_values("timestamp",ascending=False).head(5)
        st.dataframe(logs_sorted,use_container_width=True)
    else:
        st.info("No logs available yet.")

if __name__=="__main__":
    main()
