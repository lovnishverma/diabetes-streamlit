# diabetes_app.py
"""
Streamlit Diabetes Prediction App
---------------------------------
- Trains/loads a RandomForest model
- Accepts patient input for diabetes prediction
- Saves each prediction to an audit log (local + Hugging Face dataset repo)
- Displays the 5 most recent predictions
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Hugging Face utilities
from huggingface_hub import HfApi, HfFolder, Repository

# -------------------------------
# Configuration
# -------------------------------
DATA_FILE = "dia.csv"                      # Your training dataset
MODEL_FILE = "diabetes_model.pkl"          # Saved model file
SCALER_FILE = "scaler.pkl"                 # Saved scaler
LOG_DIR = Path("logs")                     # Local logs folder
LOG_FILE = LOG_DIR / "audit_log.csv"       # Local log file path

DATASET_REPO = "LovnishVerma/diabetes-logs"  # Hugging Face dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")             # Must be set in environment

LOG_DIR.mkdir(exist_ok=True)  # Ensure logs folder exists

# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------
# Model training / loading
# -------------------------------
def train_and_save_model():
    """Train a RandomForest model and save artifacts."""
    data = pd.read_csv(DATA_FILE)

    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # RandomForest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    logger.info(f"Model trained with accuracy: {acc:.2f}")
    return model, scaler


def load_model():
    """Load model and scaler if they exist, otherwise train them."""
    if Path(MODEL_FILE).exists() and Path(SCALER_FILE).exists():
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        logger.info("Loaded model and scaler from disk.")
    else:
        model, scaler = train_and_save_model()
    return model, scaler


# -------------------------------
# Logging utilities
# -------------------------------
def append_log(record: dict):
    """Append a prediction record to local + Hugging Face logs."""
    # Append to local file
    df = pd.DataFrame([record])
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
    logger.info("Prediction logged locally.")

    # Push to Hugging Face (if token available)
    if HF_TOKEN:
        try:
            repo = Repository(local_dir=LOG_DIR, clone_from=f"datasets/{DATASET_REPO}", use_auth_token=HF_TOKEN)
            repo.git_pull()
            # Copy updated log file into repo folder
            target_path = Path(repo.local_dir) / "audit_log.csv"
            df.to_csv(target_path, mode="a", header=not target_path.exists(), index=False)
            repo.push_to_hub(commit_message="Add new prediction log")
            logger.info("Prediction pushed to Hugging Face dataset.")
        except Exception as e:
            logger.warning(f"Failed to push logs to Hugging Face: {e}")
    else:
        logger.warning("HF_TOKEN not set, skipping remote log upload.")


def fetch_recent_logs():
    """Fetch logs (remote first, fallback to local). Returns a DataFrame or None."""
    logs = None
    try:
        # Raw link is confirmed working
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/audit_log.csv"
        logs = pd.read_csv(url, dtype=str)
        logger.info("Fetched logs from Hugging Face raw URL.")
    except Exception as e:
        logger.warning(f"Remote fetch failed: {e}")

    if logs is None:
        if LOG_FILE.exists():
            try:
                logs = pd.read_csv(LOG_FILE, dtype=str)
                logger.info("Loaded logs from local file.")
            except Exception as e:
                logger.warning(f"Failed reading local logs: {e}")

    return logs


# -------------------------------
# Streamlit app
# -------------------------------
def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="centered")
    st.title("🩺 Diabetes Prediction App")

    model, scaler = load_model()

    # Patient input form
    with st.form("prediction_form"):
        name = st.text_input("Patient Name", "Anonymous")
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        bloodpressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        region = st.text_input("Region", "India")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Prepare input
            features = np.array([[pregnancies, glucose, bloodpressure, skinthickness,
                                  insulin, bmi, dpf, age]])
            features_scaled = scaler.transform(features)

            # Prediction
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0][prediction]

            result = "Positive" if prediction == 1 else "Negative"
            probability = f"{proba*100:.1f}%"

            st.success(f"Prediction: {result} (Confidence: {probability})")

            # Build log record
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "name": name,
                "pregnancies": pregnancies,
                "glucose": glucose,
                "bloodpressure": bloodpressure,
                "skinthickness": skinthickness,
                "insulin": insulin,
                "bmi": bmi,
                "diabetespedigree": dpf,
                "age": age,
                "prediction": result,
                "probability": probability,
                "region": region,
            }
            append_log(record)

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            logger.exception("Prediction failed.")

    # ---------------- Show Recent Logs ----------------
    st.markdown("---")
    st.subheader("📊 Recent Predictions (Last 5)")

    logs = fetch_recent_logs()
    if logs is not None and not logs.empty:
        logs = logs.sort_values("timestamp", ascending=False).head(5)
        st.dataframe(logs)
    else:
        st.info("⚠️ No logs available yet.")


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    main()
