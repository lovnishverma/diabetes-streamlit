import streamlit as st
from joblib import load
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Global variable for model and scaler
diabetes_model = None
scaler = None

# Load the diabetes prediction model with error handling
@st.cache_resource
def load_model():
    global diabetes_model, scaler
    try:
        diabetes_model = load('models/diabetes.sav')
        # Try to load scaler if it exists
        try:
            scaler = load('models/scaler.sav')
        except FileNotFoundError:
            st.warning("Scaler not found. Using raw input values.")
            scaler = None
        return True
    except FileNotFoundError:
        st.error("❌ Model file not found at 'models/diabetes.sav'. Please ensure the model file exists.")
        return False
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return False

# Input validation functions
def validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    errors = []
    warnings = []
    
    # Critical validations (errors)
    if glucose <= 0:
        errors.append("Glucose level must be greater than 0")
    elif glucose < 70:
        warnings.append("Glucose level seems low (normal fasting: 70-100 mg/dL)")
    elif glucose > 200:
        warnings.append("Glucose level is very high - please consult a doctor immediately")
    
    if bloodpressure <= 0:
        errors.append("Blood pressure must be greater than 0")
    elif bloodpressure < 60:
        warnings.append("Blood pressure seems low (normal systolic: 90-120 mmHg)")
    elif bloodpressure > 180:
        warnings.append("Blood pressure is critically high - seek immediate medical attention")
    
    if bmi <= 0:
        errors.append("BMI must be greater than 0")
    elif bmi < 16:
        warnings.append("BMI indicates severe underweight")
    elif bmi > 40:
        warnings.append("BMI indicates severe obesity")
    
    if age <= 0:
        errors.append("Age must be greater than 0")
    
    # Logical validations
    if pregnancies > 0 and age < 12:
        errors.append("Age seems too low for the number of pregnancies")
    
    if insulin > 0 and insulin < 10:
        warnings.append("Insulin level seems unusually low")
    elif insulin > 300:
        warnings.append("Insulin level is very high")
    
    if skinthickness > 50:
        warnings.append("Skin thickness measurement seems unusually high")
    
    if diabetespedigree > 1.5:
        warnings.append("Diabetes pedigree function value is quite high")
    
    return errors, warnings

# Function to preprocess input data
def preprocess_input(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    try:
        # Create input array
        input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        
        # Apply scaling if scaler is available
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        return input_data
    except Exception as e:
        st.error(f"Error preprocessing input data: {str(e)}")
        return None

# Function to predict diabetes with probability
def predict_diabetes_with_probability(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    try:
        # Preprocess input
        input_data = preprocess_input(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age)
        
        if input_data is None:
            return None, None
        
        # Make prediction
        prediction = diabetes_model.predict(input_data)
        
        # Get probability if available
        try:
            probability = diabetes_model.predict_proba(input_data)
            prob_positive = probability[0][1] * 100  # Probability of positive class
        except AttributeError:
            # Model doesn't support predict_proba
            prob_positive = None
        
        return bool(prediction[0]), prob_positive
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Function to display logo with error handling
def display_logo():
    logo_path = "static/logo.png"
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load logo: {str(e)}")
    else:
        # Display a simple text logo instead
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: #1f77b4;'>🩺 DIABETES PREDICTION</h2>
        </div>
        """, unsafe_allow_html=True)

# Function to display results with enhanced formatting
def display_results(name, prediction, probability, errors, warnings):
    if prediction is None:
        st.error("❌ Could not make prediction due to errors.")
        return
    
    # Display warnings if any
    if warnings:
        st.warning("⚠️ **Please note the following:**")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    # Display result
    if prediction:
        st.error(f"🔴 **Hello {name}, your diabetes risk assessment is: POSITIVE**")
        if probability:
            st.error(f"**Risk Probability: {probability:.1f}%**")
        st.markdown("""
        **⚠️ IMPORTANT:** This is a preliminary assessment. Please consult with a healthcare professional 
        for proper medical diagnosis and treatment.
        """)
    else:
        st.success(f"✅ **Hello {name}, your diabetes risk assessment is: NEGATIVE**")
        if probability:
            st.success(f"**Risk Probability: {probability:.1f}%**")
        st.markdown("""
        **ℹ️ NOTE:** This indicates lower risk based on the provided parameters. Continue maintaining 
        a healthy lifestyle and regular medical check-ups.
        """)

# Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Diabetes Prediction App",
        page_icon="💉",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # Load model
    if not load_model():
        st.stop()
    
    # Header
    st.title("🩺 Diabetes Risk Assessment Tool")
    display_logo()
    
    st.markdown("""
    This tool uses machine learning to assess diabetes risk based on clinical parameters. 
    **Please note:** This is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Input form
    st.sidebar.header("📋 Patient Information")
    
    # User details
    name = st.sidebar.text_input("👤 Full Name", placeholder="Enter patient name", help="Enter the patient's full name")
    
    if not name:
        st.info("👈 Please enter patient details in the sidebar to begin assessment.")
        return
    
    # Clinical details
    st.sidebar.subheader("🔬 Clinical Parameters")
    
    # Create two columns for better organization
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        pregnancies = st.number_input("🤱 Pregnancies", min_value=0, max_value=20, value=0, 
                                    help="Number of times pregnant")
        glucose = st.number_input("🍯 Glucose (mg/dL)", min_value=0, max_value=300, value=120, 
                                help="Plasma glucose concentration (normal fasting: 70-100)")
        bloodpressure = st.number_input("💓 Blood Pressure (mmHg)", min_value=0, max_value=200, value=80, 
                                      help="Diastolic blood pressure (normal: 60-80)")
        insulin = st.number_input("💉 Insulin (mu U/ml)", min_value=0, max_value=500, value=0, 
                                help="2-Hour serum insulin")
    
    with col2:
        bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=50.0, value=25.0, format="%.1f",
                            help="Body mass index (normal: 18.5-24.9)")
        diabetespedigree = st.number_input("🧬 Diabetes Pedigree", min_value=0.0, max_value=2.0, value=0.5, format="%.3f",
                                         help="Diabetes pedigree function (family history)")
        age = st.number_input("📅 Age (years)", min_value=1, max_value=100, value=30, 
                            help="Age in years")
        skinthickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20, 
                                      help="Triceps skin fold thickness")
    
    # Add reference ranges
    st.sidebar.markdown("""
    **📊 Reference Ranges:**
    - Glucose (fasting): 70-100 mg/dL
    - Blood Pressure: 60-80 mmHg (diastolic)
    - BMI: 18.5-24.9 (normal)
    - Age: Consider diabetes risk increases with age
    """)
    
    # Prediction button
    if st.sidebar.button("🔍 Assess Diabetes Risk", key="predict_button"):
        # Validate inputs
        errors, warnings = validate_inputs(pregnancies, glucose, bloodpressure, skinthickness, 
                                         insulin, bmi, diabetespedigree, age)
        
        if errors:
            st.error("❌ **Input Validation Errors:**")
            for error in errors:
                st.error(f"• {error}")
            st.info("Please correct the errors above and try again.")
            return
        
        # Make prediction
        with st.spinner("🔄 Analyzing patient data..."):
            prediction, probability = predict_diabetes_with_probability(
                pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age
            )
        
        # Display results
        display_results(name, prediction, probability, errors, warnings)
        
        # Display input summary
        with st.expander("📋 Input Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Pregnancies:** {pregnancies}")
                st.write(f"**Glucose:** {glucose} mg/dL")
                st.write(f"**Blood Pressure:** {bloodpressure} mmHg")
                st.write(f"**Insulin:** {insulin} mu U/ml")
            with col2:
                st.write(f"**BMI:** {bmi:.1f}")
                st.write(f"**Diabetes Pedigree:** {diabetespedigree:.3f}")
                st.write(f"**Age:** {age} years")
                st.write(f"**Skin Thickness:** {skinthickness} mm")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **⚠️ Medical Disclaimer:**
    This tool is for educational purposes only. 
    Always consult healthcare professionals for medical advice.
    
    **📞 Emergency:** If you have severe symptoms, seek immediate medical attention.
    
    © 2024 Diabetes Prediction App
    """)
    
    # Additional information in main area
    if name:
        st.markdown("---")
        st.subheader("📚 About Diabetes Risk Factors")
        
        tab1, tab2, tab3 = st.tabs(["🔍 Risk Factors", "📊 Understanding Results", "🏥 Next Steps"])
        
        with tab1:
            st.markdown("""
            **Key Risk Factors for Diabetes:**
            - **Age:** Risk increases with age, especially after 45
            - **BMI:** Higher BMI indicates increased risk
            - **Family History:** Genetic predisposition (pedigree function)
            - **Blood Glucose:** Elevated glucose levels
            - **Blood Pressure:** Hypertension is linked to diabetes risk
            - **Previous Pregnancies:** Gestational diabetes history
            """)
        
        with tab2:
            st.markdown("""
            **Understanding Your Results:**
            - **NEGATIVE:** Lower risk based on current parameters
            - **POSITIVE:** Higher risk - requires medical evaluation
            - **Probability Score:** When available, shows percentage risk
            
            **Important:** This is a screening tool, not a diagnosis.
            """)
        
        with tab3:
            st.markdown("""
            **Recommended Next Steps:**
            
            **If Result is POSITIVE:**
            - 🏥 Schedule appointment with healthcare provider
            - 🩸 Request proper diagnostic tests (HbA1c, OGTT)
            - 📝 Discuss family history and symptoms
            
            **If Result is NEGATIVE:**
            - 🔄 Continue regular health check-ups
            - 🥗 Maintain healthy diet and exercise
            - 📅 Rescreen annually if risk factors present
            """)

# Run the Streamlit app
if __name__ == "__main__":
    main()