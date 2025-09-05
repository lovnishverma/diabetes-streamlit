"""
Streamlit diabetes risk app (Production Version)

- Fetches model artifacts from models/
- Predicts diabetes risk using a cleaner, form-based UI.
- Logs each prediction by fetching the current audit_log.csv from HF,
  appending the new row, and re-uploading the merged CSV.
  This avoids overwrites and preserves history across restarts.
- Enhanced for robustness, maintainability, and user experience.
"""

import os
import time
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from huggingface_hub import HfApi, hf_hub_download, create_repo

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# Centralized config for easier management and deployment.
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Main app configuration
CONFIG = {
    "model_dir": Path("models"),
    "model_file": "diabetes.sav",
    "scaler_file": "scaler.sav",
    "medians_file": "medians.sav",
    "hf_username": "LovnishVerma",
    "hf_repo_name": "diabetes-logs",
    "hf_token": os.getenv("HF_TOKEN"),
    "log_filename": "audit_log.csv",
    "required_median_keys": {
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    }
}
CONFIG["dataset_repo_id"] = f"{CONFIG['hf_username']}/{CONFIG['hf_repo_name']}"


# -----------------------------------------------------------------------------
# 2. HUGGING FACE UTILITIES
# Functions for interacting with the Hugging Face Hub dataset repo.
# -----------------------------------------------------------------------------
def ensure_dataset_repo():
    """Create dataset repo on Hugging Face (no-op if it exists)."""
    try:
        create_repo(
            repo_id=CONFIG["dataset_repo_id"],
            token=CONFIG["hf_token"],
            private=False,
            repo_type="dataset",
            exist_ok=True,
        )
    except Exception as e:
        logger.info(f"Could not create or verify repo (might already exist): {e}")

def fetch_remote_logs(retries: int = 2, delay: float = 0.5) -> pd.DataFrame:
    """Robustly fetch the remote audit log CSV from the HF dataset."""
    if not CONFIG["hf_token"]:
        logger.warning("HF_TOKEN not found. Cannot fetch remote logs.")
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            local_path = hf_hub_download(
                repo_id=CONFIG["dataset_repo_id"],
                filename=CONFIG["log_filename"],
                repo_type="dataset",
                token=CONFIG["hf_token"],
            )
            df = pd.read_csv(local_path, dtype=str)
            logger.info("Successfully fetched remote logs using hf_hub_download.")
            return df
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} to fetch logs failed: {e}")
            time.sleep(delay)
    
    logger.error("All attempts to fetch remote logs failed.")
    return pd.DataFrame()

def upload_merged_logs(local_csv_path: str):
    """Upload a local CSV file to the HF dataset, overwriting the remote file."""
    if not CONFIG["hf_token"]:
        st.error("`HF_TOKEN` is not set. Cannot upload logs.")
        raise ValueError("HF_TOKEN is required for uploading logs.")
    
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_csv_path,
            path_in_repo=CONFIG["log_filename"],
            repo_id=CONFIG["dataset_repo_id"],
            repo_type="dataset",
            token=CONFIG["hf_token"],
            commit_message=f"Update audit log from app: {datetime.utcnow().isoformat()}",
        )
        logger.info("Successfully uploaded merged audit log to Hugging Face.")
    except Exception as e:
        logger.exception("Failed to upload merged logs to Hugging Face.")
        st.error(f"Error uploading logs to Hugging Face Hub. Please ensure your token has 'write' permissions. Details: {e}")
        raise

# -----------------------------------------------------------------------------
# 3. MODEL & PREDICTION LOGIC
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources() -> Tuple[Optional[Any], Optional[Any], Optional[Dict]]:
    """Load model, scaler, and medians from disk. Cached for performance."""
    try:
        model_path = CONFIG["model_dir"] / CONFIG["model_file"]
        scaler_path = CONFIG["model_dir"] / CONFIG["scaler_file"]
        medians_path = CONFIG["model_dir"] / CONFIG["medians_file"]

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        medians = joblib.load(medians_path)

        if not CONFIG["required_median_keys"].issubset(medians.keys()):
            raise ValueError("Medians object is missing required keys.")

        logger.info("Model, scaler, and medians loaded successfully.")
        return model, scaler, medians
    except FileNotFoundError as e:
        st.error(f"**Error:** A required model file is missing: `{e.filename}`. Please ensure `models/` contains all necessary files.")
        logger.error(f"Missing model artifact: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model resources: {e}")
        logger.exception("load_resources failed unexpectedly.")
        return None, None, None

def validate_inputs(inputs: Dict[str, Any]) -> List[str]:
    """Validate user inputs against sensible clinical and logical constraints."""
    errors = []
    if not (0 < inputs["glucose"] <= 300): errors.append("Glucose must be between 1 and 300 mg/dL.")
    if not (0 < inputs["bloodpressure"] <= 200): errors.append("Blood Pressure must be between 1 and 200 mmHg.")
    if not (0 < inputs["bmi"] <= 70): errors.append("BMI must be between 1 and 70.")
    if not (1 <= inputs["age"] <= 120): errors.append("Age must be between 1 and 120.")
    if inputs["pregnancies"] < 0 or inputs["pregnancies"] > 20: errors.append("Pregnancies must be between 0 and 20.")
    if inputs["age"] < 15 and inputs["pregnancies"] > 0: errors.append("Age is too low for the number of pregnancies reported.")
    return errors

def make_prediction(model: Any, scaler: Any, medians: Dict, inputs: Dict[str, Any]) -> Tuple[Optional[bool], Optional[float]]:
    """Preprocess inputs and return a prediction and probability."""
    try:
        df = pd.DataFrame([inputs])
        
        # Replace 0s with NaN for columns where 0 is a placeholder for missing data
        zero_placeholder_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[zero_placeholder_cols] = df[zero_placeholder_cols].replace(0, np.nan)
        
        # Impute missing values with medians
        df.fillna(medians, inplace=True)
        
        # Scale the features and predict
        scaled_features = scaler.transform(df)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] * 100
        
        return bool(prediction), float(probability)
    except Exception as e:
        logger.exception("Prediction failed.")
        st.error(f"Prediction calculation failed: {e}")
        return None, None

# -----------------------------------------------------------------------------
# 4. LOGGING WORKFLOW
# The core fetch -> append -> upload logic for audit trails.
# -----------------------------------------------------------------------------
def log_prediction(name: str, inputs: Dict, prediction: bool, probability: float):
    """Fetches remote logs, appends the new prediction, and uploads the merged file."""
    if not CONFIG["hf_token"]:
        logger.warning("HF_TOKEN not set. Skipping remote logging.")
        return
        
    try:
        remote_logs_df = fetch_remote_logs()
        
        new_log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name or "Anonymous",
            "prediction": "High Risk" if prediction else "Low Risk",
            "probability_percent": f"{probability:.1f}",
            "region": "India", # Example of a static field
            **inputs  # Unpack all user inputs into columns
        }

        new_log_df = pd.DataFrame([new_log_entry])
        merged_df = pd.concat([remote_logs_df, new_log_df], ignore_index=True)
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            merged_df.to_csv(tmp.name, index=False)
            upload_merged_logs(tmp.name)
        
        os.unlink(tmp.name) # Clean up the temp file
        st.toast("Prediction logged successfully!", icon="📝")

    except Exception as e:
        logger.exception("Failed to log prediction.")
        st.warning("Could not save the prediction log due to an error.", icon="⚠️")

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI COMPONENTS
# Breaking the UI into functions for clarity and reusability.
# -----------------------------------------------------------------------------
def render_header():
    """Renders the main title and introduction for the app."""
    st.title("🩺 Diabetes Risk Assessment")
    st.markdown("""
    This app predicts the risk of diabetes based on key health indicators. 
    Please enter the patient's information below. All values should be based on a recent medical check-up.
    """)
    st.info("**Disclaimer:** This is a demonstration tool and not a substitute for professional medical advice.", icon="ℹ️")

def render_input_form() -> Optional[Dict[str, Any]]:
    """Displays the input form and returns a dictionary of values upon submission."""
    with st.form("prediction_form"):
        st.header("Patient Information")
        
        name = st.text_input("Name (Optional)")
        
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, value=1, help="Number of times pregnant.")
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, value=120, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test.")
            bloodpressure = st.number_input("Blood Pressure (mmHg)", 0, 200, value=80, help="Diastolic blood pressure.")
            skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, value=20, help="Triceps skin fold thickness.")
        
        with col2:
            insulin = st.number_input("Insulin (μU/mL)", 0, 900, value=0, help="2-Hour serum insulin.")
            bmi = st.number_input("BMI", 0.0, 70.0, value=25.0, format="%.1f", help="Body Mass Index (weight in kg / (height in m)^2).")
            diabetespedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, value=0.47, format="%.3f", help="A function that scores likelihood of diabetes based on family history.")
            age = st.number_input("Age", 1, 120, value=30, help="Patient's age in years.")
        
        submitted = st.form_submit_button("Assess Risk", type="primary")

        if submitted:
            return {
                "name": name,
                "pregnancies": pregnancies, "glucose": glucose, "bloodpressure": bloodpressure,
                "skinthickness": skinthickness, "insulin": insulin, "bmi": bmi,
                "diabetespedigree": diabetespedigree, "age": age
            }
    return None

def render_results():
    """Displays the prediction result stored in the session state."""
    if "prediction" in st.session_state:
        pred = st.session_state.prediction
        prob = st.session_state.probability
        
        st.header("Assessment Result")
        
        if pred:
            st.error(f"**Result: High Risk of Diabetes**", icon="🔴")
        else:
            st.success(f"**Result: Low Risk of Diabetes**", icon="✅")

        st.metric(label="Predicted Risk Probability", value=f"{prob:.1f}%")
        st.progress(int(prob))
        
        with st.expander("View Submitted Data"):
            st.json(st.session_state.inputs)

def render_recent_logs():
    """Fetches and displays the last 5 prediction logs."""
    st.markdown("---")
    st.subheader("Recent Predictions")
    try:
        logs_df = fetch_remote_logs()
        if not logs_df.empty and "timestamp" in logs_df.columns:
            display_cols = ["timestamp", "name", "prediction", "probability_percent", "age", "glucose", "bmi"]
            # Filter to columns that exist in the dataframe to avoid errors
            cols_to_show = [col for col in display_cols if col in logs_df.columns]
            st.dataframe(
                logs_df.sort_values("timestamp", ascending=False).head(5)[cols_to_show],
                hide_index=True
            )
        else:
            st.info("No recent prediction logs found.")
    except Exception as e:
        logger.error(f"Failed to display recent logs: {e}")
        st.warning("Could not fetch recent logs at this time.")

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION FLOW
# -----------------------------------------------------------------------------
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

    # One-time setup
    if "repo_setup" not in st.session_state:
        if CONFIG["hf_token"]:
            ensure_dataset_repo()
        st.session_state.repo_setup = True

    model, scaler, medians = load_resources()
    if not all([model, scaler, medians]):
        st.stop() # Stop execution if resources failed to load

    render_header()
    
    if not CONFIG["hf_token"]:
        st.warning("`HF_TOKEN` environment variable not found. Prediction logging is disabled.", icon="⚠️")
        
    form_data = render_input_form()

    if form_data:
        # Extract name and numerical inputs
        name = form_data.pop("name")
        inputs = {k: v for k, v in form_data.items() if k != 'name'}
        
        # 1. Validate inputs
        errors = validate_inputs(inputs)
        if errors:
            for error in errors:
                st.error(error)
        else:
            # 2. Predict and Log
            with st.spinner("Analyzing data and logging prediction..."):
                pred, prob = make_prediction(model, scaler, medians, inputs)
                
                if pred is not None:
                    # Store results in session state for display
                    st.session_state.prediction = pred
                    st.session_state.probability = prob
                    st.session_state.inputs = inputs
                    
                    log_prediction(name, inputs, pred, prob)

    # 3. Display results and logs
    render_results()
    render_recent_logs()

if __name__ == "__main__":
    main()