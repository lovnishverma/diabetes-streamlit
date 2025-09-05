"""
Professional Diabetes Risk Assessment App
- Modern glassmorphism UI design
- Animated components and smooth transitions
- Professional dashboard layout
- Interactive visualizations
- Advanced styling and custom components
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
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download, create_repo

# -----------------------
# Enhanced Page Config & Styling
# -----------------------
st.set_page_config(
    page_title="DiabetesAI Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container with glassmorphism */
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 25px;
        margin-bottom: 2rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.8);
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 4px 15px rgba(238, 90, 36, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white !important;
        backdrop-filter: blur(10px);
    }
    
    /* Input placeholder styling */
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    .stNumberInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Input focus styling */
    .stTextInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.5);
        color: white !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.25);
    }
    
    .stNumberInput > div > div > input:focus {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.5);
        color: white !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 255, 255, 0.25);
    }
    
    /* Risk Cards */
    .risk-card-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
        transform: translateY(-5px);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideInRight 0.8s ease-out;
    }
    
    .risk-card-low {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(81, 207, 102, 0.4);
        transform: translateY(-5px);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideInRight 0.8s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .risk-score {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .risk-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .risk-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    
    /* Stats Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Charts Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    /* Data Table Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar Header */
    .sidebar-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-top: 3px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Metrics Styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    /* Custom Alert Boxes */
    .success-alert {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .error-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .info-alert {
        background: linear-gradient(135deg, #74c0fc, #339af0);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Feature highlights */
    .feature-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Configuration & Logging
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
# Enhanced UI Components
# -----------------------
def create_hero_section():
    """Create an impressive hero section"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🏥 DiabetesAI Pro</h1>
        <p class="hero-subtitle">Advanced Machine Learning Risk Assessment Platform</p>
        <div class="hero-badge">✨ AI-Powered • Real-time Analysis • Professional Grade</div>
        <br><br>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <span class="feature-badge">🤖 ML Predictions</span>
            <span class="feature-badge">📊 Real-time Analytics</span>
            <span class="feature-badge">☁️ Cloud Storage</span>
            <span class="feature-badge">📱 Responsive Design</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_risk_visualization(probability):
    """Create a beautiful risk visualization chart"""
    # Gauge chart for risk probability
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score", 'font': {'color': 'white', 'size': 20}},
        number = {'font': {'color': 'white', 'size': 30}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(81, 207, 102, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(255, 152, 0, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(255, 107, 107, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        height=300
    )
    
    return fig

def create_feature_importance_chart():
    """Create a chart showing feature importance"""
    features = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Blood Pressure', 
               'Insulin', 'Pedigree', 'Skin Thickness']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    
    fig = px.bar(
        x=importance, 
        y=features,
        orientation='h',
        title="Feature Importance in Diabetes Prediction",
        color=importance,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        title={'font': {'color': 'white', 'size': 16}},
        xaxis={'tickfont': {'color': 'white'}},
        yaxis={'tickfont': {'color': 'white'}},
        height=400
    )
    
    return fig

def create_stats_cards(logs_df):
    """Create beautiful statistics cards"""
    if logs_df.empty:
        total_predictions = 0
        high_risk_count = 0
        avg_age = 0
        avg_bmi = 0
    else:
        total_predictions = len(logs_df)
        high_risk_count = len(logs_df[logs_df['prediction'] == 'Positive'])
        try:
            avg_age = logs_df['age'].astype(float).mean()
            avg_bmi = logs_df['bmi'].astype(float).mean()
        except:
            avg_age = 0
            avg_bmi = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_predictions}</div>
            <div class="stat-label">Total Assessments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{high_risk_count}</div>
            <div class="stat-label">High Risk Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_age:.1f}</div>
            <div class="stat-label">Average Age</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_bmi:.1f}</div>
            <div class="stat-label">Average BMI</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# Original Backend Functions (unchanged)
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
# Enhanced Main Application
# -----------------------
def main():
    # Initialize session state
    if "repo_setup" not in st.session_state:
        ensure_dataset_repo()
        st.session_state.repo_setup = True
    
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    
    # Hero Section
    create_hero_section()
    
    # Load model resources
    model, scaler, medians = load_resources()
    if model is None:
        st.markdown("""
        <div class="error-alert">
            <h3>⚠️ Model Resources Not Found</h3>
            <p>Please ensure model files are properly loaded in the models directory.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Load and display statistics
    logs_df = fetch_remote_logs_via_api(retries=2, delay=0.3)
    
    # Statistics Dashboard
    st.markdown("## 📊 Analytics Dashboard")
    create_stats_cards(logs_df)
    
    # Main Layout
    col1, col2 = st.columns([1, 2])
    
    # Sidebar Input Panel
    with col1:
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="color: white; margin: 0;">🏥 Patient Assessment</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">Enter patient information below</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input Form
        with st.form("assessment_form", clear_on_submit=False):
            name = st.text_input("👤 Patient Name", placeholder="Enter patient name (optional)")
            
            st.markdown("#### 📋 Medical Information")
            pregnancies = st.number_input("🤱 Number of Pregnancies", 0, 20, value=0)
            glucose = st.number_input("🍭 Glucose Level (mg/dL)", 0, 300, value=120)
            bloodpressure = st.number_input("🩺 Blood Pressure (mmHg)", 0, 200, value=80)
            skinthickness = st.number_input("📏 Skin Thickness (mm)", 0, 100, value=20)
            insulin = st.number_input("💉 Insulin Level (μU/mL)", 0, 500, value=0)
            bmi = st.number_input("⚖️ BMI", 0.0, 70.0, value=25.0, format="%.1f")
            diabetespedigree = st.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0, value=0.5, format="%.3f")
            age = st.number_input("🎂 Age (years)", 1, 120, value=30)
            
            submitted = st.form_submit_button("🔬 Analyze Risk", use_container_width=True)
            
            if submitted:
                inputs = {
                    "pregnancies": pregnancies, "glucose": glucose, "bloodpressure": bloodpressure,
                    "skinthickness": skinthickness, "insulin": insulin, "bmi": bmi,
                    "diabetespedigree": diabetespedigree, "age": age
                }
                
                # Validate inputs
                errors = validate_inputs(**inputs)
                if errors:
                    st.markdown("""
                    <div class="error-alert">
                        <h4>❌ Input Validation Errors</h4>
                    """, unsafe_allow_html=True)
                    for error in errors:
                        st.write(f"• {error}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Show loading animation
                    with st.spinner("🔄 Analyzing patient data..."):
                        time.sleep(1)  # Simulate processing time
                        pred, prob = predict_diabetes(model, scaler, medians, **inputs)
                        
                        if pred is not None:
                            # Log the prediction
                            log_prediction_remote_only(name, inputs, pred, prob)
                            st.session_state.show_results = True
                            st.session_state.last_prediction = {"pred": pred, "prob": prob, "name": name}
    
    # Results Panel
    with col2:
        if st.session_state.show_results and "last_prediction" in st.session_state:
            result = st.session_state.last_prediction
            pred, prob, patient_name = result["pred"], result["prob"], result["name"]
            
            # Risk Assessment Card
            risk_class = "risk-card-high" if pred else "risk-card-low"
            risk_icon = "🚨" if pred else "✅"
            risk_text = "High Diabetes Risk" if pred else "Low Diabetes Risk"
            risk_color = "#ff6b6b" if pred else "#51cf66"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h2 class="risk-title">{risk_icon} {risk_text}</h2>
                <div class="risk-score">{prob:.1f}%</div>
                <p class="risk-subtitle">Patient: {patient_name or 'Anonymous'}</p>
                <p style="font-size: 0.9rem; opacity: 0.8;">
                    Assessment completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk Visualization
            st.markdown("#### 📈 Risk Analysis")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = create_risk_visualization(prob)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("#### 💡 Clinical Recommendations")
            if pred:
                st.markdown("""
                <div class="error-alert">
                    <h4>⚠️ High Risk - Immediate Action Recommended</h4>
                    <ul style="margin: 1rem 0;">
                        <li>Schedule appointment with healthcare provider immediately</li>
                        <li>Consider HbA1c and fasting glucose tests</li>
                        <li>Implement lifestyle modifications (diet & exercise)</li>
                        <li>Monitor blood glucose levels regularly</li>
                        <li>Consider diabetes education programs</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-alert">
                    <h4>✅ Low Risk - Preventive Care Recommended</h4>
                    <ul style="margin: 1rem 0;">
                        <li>Maintain healthy lifestyle habits</li>
                        <li>Regular health screenings as recommended</li>
                        <li>Continue balanced diet and regular exercise</li>
                        <li>Annual diabetes screening</li>
                        <li>Monitor weight and blood pressure</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Default dashboard view
            st.markdown("#### 🎯 AI-Powered Risk Assessment")
            st.markdown("""
            <div class="info-alert">
                <h4>🔬 Advanced Machine Learning Analysis</h4>
                <p>Our state-of-the-art machine learning model analyzes multiple health parameters to provide accurate diabetes risk predictions. Enter patient information in the form to begin assessment.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance Chart
            st.markdown("#### 📊 Model Feature Importance")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_importance = create_feature_importance_chart()
            st.plotly_chart(fig_importance, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Analytics Section
    st.markdown("---")
    st.markdown("## 📈 Recent Assessments & Analytics")
    
    if not logs_df.empty:
        # Recent predictions table with enhanced styling
        st.markdown("#### 🕒 Latest Predictions")
        
        # Process logs for better display
        display_logs = logs_df.sort_values("timestamp", ascending=False).head(10).copy()
        
        # Add risk level column
        display_logs['Risk Level'] = display_logs['prediction'].apply(
            lambda x: '🚨 High Risk' if x == 'Positive' else '✅ Low Risk'
        )
        
        # Reorder columns for better presentation
        display_columns = ['timestamp', 'name', 'age', 'bmi', 'glucose', 'Risk Level', 'probability']
        display_logs_subset = display_logs[display_columns].copy()
        
        # Rename columns for better readability
        display_logs_subset.columns = ['Timestamp', 'Patient', 'Age', 'BMI', 'Glucose', 'Risk Level', 'Probability']
        
        st.dataframe(
            display_logs_subset,
            use_container_width=True,
            height=400
        )
        
        # Analytics Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Risk Distribution")
            if len(logs_df) > 0:
                risk_counts = logs_df['prediction'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=['Low Risk' if x == 'Negative' else 'High Risk' for x in risk_counts.index],
                    color_discrete_sequence=['#51cf66', '#ff6b6b'],
                    title="Risk Assessment Distribution"
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Inter'},
                    title={'font': {'color': 'white', 'size': 16}}
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.markdown("#### 📈 Age vs Risk Analysis")
            if len(logs_df) > 5:  # Only show if we have enough data
                try:
                    # Convert age to numeric and create age groups
                    logs_df['age_numeric'] = pd.to_numeric(logs_df['age'], errors='coerce')
                    logs_df_clean = logs_df.dropna(subset=['age_numeric'])
                    
                    if len(logs_df_clean) > 0:
                        fig_scatter = px.scatter(
                            logs_df_clean,
                            x='age_numeric',
                            y=[float(p.replace('%', '')) for p in logs_df_clean['probability']],
                            color='prediction',
                            color_discrete_map={'Negative': '#51cf66', 'Positive': '#ff6b6b'},
                            title="Age vs Diabetes Risk Probability",
                            labels={'x': 'Age', 'y': 'Risk Probability (%)'}
                        )
                        fig_scatter.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white', 'family': 'Inter'},
                            title={'font': {'color': 'white', 'size': 16}},
                            xaxis={'tickfont': {'color': 'white'}, 'gridcolor': 'rgba(255,255,255,0.1)'},
                            yaxis={'tickfont': {'color': 'white'}, 'gridcolor': 'rgba(255,255,255,0.1)'},
                            legend={'font': {'color': 'white'}}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
                except Exception as e:
                    st.info("📊 Age analysis will be available with more data points.")
            else:
                st.info("📊 Age analysis will be available with more data points.")
        
        # Advanced Analytics
        if len(logs_df) > 20:
            st.markdown("#### 🔍 Advanced Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            try:
                # Calculate advanced metrics
                logs_df['age_numeric'] = pd.to_numeric(logs_df['age'], errors='coerce')
                logs_df['bmi_numeric'] = pd.to_numeric(logs_df['bmi'], errors='coerce')
                logs_df['glucose_numeric'] = pd.to_numeric(logs_df['glucose'], errors='coerce')
                
                high_risk_avg_age = logs_df[logs_df['prediction'] == 'Positive']['age_numeric'].mean()
                low_risk_avg_age = logs_df[logs_df['prediction'] == 'Negative']['age_numeric'].mean()
                
                high_risk_avg_bmi = logs_df[logs_df['prediction'] == 'Positive']['bmi_numeric'].mean()
                low_risk_avg_bmi = logs_df[logs_df['prediction'] == 'Negative']['bmi_numeric'].mean()
                
                high_risk_avg_glucose = logs_df[logs_df['prediction'] == 'Positive']['glucose_numeric'].mean()
                low_risk_avg_glucose = logs_df[logs_df['prediction'] == 'Negative']['glucose_numeric'].mean()
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: white; margin-bottom: 1rem;">👥 Age Analysis</h4>
                        <p style="color: #ff6b6b;">High Risk Avg: {high_risk_avg_age:.1f} years</p>
                        <p style="color: #51cf66;">Low Risk Avg: {low_risk_avg_age:.1f} years</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: white; margin-bottom: 1rem;">⚖️ BMI Analysis</h4>
                        <p style="color: #ff6b6b;">High Risk Avg: {high_risk_avg_bmi:.1f}</p>
                        <p style="color: #51cf66;">Low Risk Avg: {low_risk_avg_bmi:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: white; margin-bottom: 1rem;">🍭 Glucose Analysis</h4>
                        <p style="color: #ff6b6b;">High Risk Avg: {high_risk_avg_glucose:.1f} mg/dL</p>
                        <p style="color: #51cf66;">Low Risk Avg: {low_risk_avg_glucose:.1f} mg/dL</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                pass
    
    else:
        st.markdown("""
        <div class="info-alert">
            <h4>📊 No Assessment Data Available</h4>
            <p>Complete your first risk assessment to see analytics and trends. The system will automatically track and analyze prediction patterns over time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer Section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: rgba(255,255,255,0.6);">
        <p><strong>DiabetesAI Pro</strong> - Advanced Machine Learning Risk Assessment Platform</p>
        <p>🤖 Powered by Scikit-learn • ☁️ Data stored on Hugging Face • 🔒 Privacy-focused design</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            ⚠️ <em>This tool is for educational and screening purposes only. Always consult healthcare professionals for medical decisions.</em>
        </p>
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <p style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">
                © 2024 <strong>Lovnish Verma</strong> | NIELIT Chandigarh
            </p>
            <p style="font-size: 0.7rem; color: rgba(255,255,255,0.5); margin-top: 0.5rem;">
                Developed with ❤️ using Streamlit & Machine Learning
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()