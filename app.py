import streamlit as st
import numpy as np
import joblib
from PIL import Image
import base64
import os

# Load models and scalers
bank_model = joblib.load("bank_churn_model.pkl")
telecom_model = joblib.load("telecom_churn_model.pkl")
bank_scaler = joblib.load("scaler_bank.pkl")
telecom_scaler = joblib.load("scaler_telecom.pkl")

# Function to predict churn
def predict_churn(model, scaler, features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return "Churned" if prediction[0] == 1 else "Not Churned"

# Streamlit UI
st.set_page_config(page_title="Churn Prediction App", page_icon="🔍", layout="wide")

# Function to set video background
def set_video_background(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}
        .video-background {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            opacity: 0.6;
        }}
        </style>
        <video autoplay muted loop class="video-background">
            <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# Set video background from assets folder
video_path = os.path.join("assets", "background.mp4")  # Update with your video filename
try:
    set_video_background(video_path)
except Exception as e:
    st.warning(f"Could not load background video: {e}")
    # Fallback to a color background
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1c3b5a, #0a192f);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom CSS for advanced form styling
st.markdown(
    """
    <style>
        /* Main app styling */
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Main title styling - ENHANCED */
        .main-title {
            text-align: center;
            font-size: 3.8rem;
            font-weight: 700;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.9);
            margin: 1.5rem 0;
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 25px;
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            animation: shimmer 3s ease-in-out infinite alternate;
        }
        
        @keyframes shimmer {
            from {
                text-shadow: 0 0 5px #fff, 0 0 10px #00C9FF, 0 0 15px #00C9FF;
            }
            to {
                text-shadow: 0 0 10px #fff, 0 0 20px #92FE9D, 0 0 30px #92FE9D;
            }
        }
        
        /* Form container styling - ENHANCED with new color scheme */
        .form-container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
            margin: 20px 0;
            border-left: 6px solid #00C9FF;
            transition: all 0.3s ease;
        }
        
        .form-container:hover {
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.45);
            transform: translateY(-5px);
        }
        
        /* Section headers - ENHANCED */
        .section-header {
            color: #1E3A8A;
            font-size: 2.2rem;
            font-weight: 600;
            margin-bottom: 25px;
            border-bottom: 3px solid #00C9FF;
            padding-bottom: 15px;
            text-align: center;
            background: linear-gradient(90deg, #1E3A8A, #0284C7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Input field styling - ENHANCED */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input {
            border-radius: 12px;
            padding: 14px 18px;
            border: 2px solid #ddd;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 12px;
            font-size: 16px;
            width: 100%;
            background-color: #F8FAFC;
        }
        
        .stTextInput>div>div>input:focus, 
        .stNumberInput>div>div>input:focus {
            border-color: #00C9FF;
            box-shadow: 0 0 0 3px rgba(0, 201, 255, 0.2);
            transform: translateY(-2px);
            background-color: #FFFFFF;
        }
        
        /* Label styling - ENHANCED */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .css-16huue1 {
            font-weight: 600 !important;
            font-size: 17px !important;
            color: #1E3A8A !important;
            margin-bottom: 10px !important;
            letter-spacing: 0.2px !important;
        }
        
        /* Input field container - ENHANCED */
        .input-container {
            margin-bottom: 28px;
            border-bottom: 1px dashed #E2E8F0;
            padding-bottom: 18px;
            transition: all 0.3s ease;
        }
        
        .input-container:hover {
            border-bottom-color: #00C9FF;
            background-color: rgba(0, 201, 255, 0.03);
            border-radius: 8px;
            padding: 5px;
        }
        
        /* Select box styling - ENHANCED */
        .stSelectbox>div>div {
            border-radius: 12px !important;
            border: 2px solid #E2E8F0 !important;
            margin-bottom: 12px;
            background-color: #F8FAFC;
        }
        
        .stSelectbox>div>div:focus-within {
            border-color: #00C9FF !important;
            box-shadow: 0 0 0 3px rgba(0, 201, 255, 0.2) !important;
            background-color: #FFFFFF;
        }
        
        /* Radio button styling - ENHANCED */
        .stRadio>div {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            border: 1px solid #E2E8F0;
        }
        
        .stRadio>div:hover {
            border-color: #00C9FF;
            background-color: rgba(0, 201, 255, 0.05);
        }
        
        /* Slider styling - ENHANCED */
        .stSlider>div>div>div>div {
            background-color: #00C9FF !important;
        }
        
        /* Button styling - ENHANCED */
        .stButton>button {
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            color: #1E293B;
            font-weight: 700;
            border-radius: 12px;
            padding: 15px 28px;
            border: none;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-size: 16px;
            width: 100%;
            margin-top: 18px;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
            background: linear-gradient(90deg, #00C9FF, #92FE9D);
            color: #0F172A;
        }
        
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Success message styling - ENHANCED */
        .element-container div[data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 22px;
            margin: 22px 0;
            animation: fadeIn 0.7s ease-in-out;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(25px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Sidebar styling - ENHANCED */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(23, 37, 84, 0.9)) !important;
            box-shadow: 3px 0 20px rgba(0, 0, 0, 0.4);
        }
        
        [data-testid="stSidebar"] .block-container {
            padding: 2.8rem 1.7rem;
        }
        
        /* Sidebar title - NEW */
        [data-testid="stSidebar"] h2 {
            color: #E0F2FE !important;
            font-size: 1.8rem !important;
            margin-bottom: 25px !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
            letter-spacing: 0.5px !important;
            border-bottom: 2px solid #00C9FF !important;
            padding-bottom: 10px !important;
        }
        
        /* Sidebar text - NEW */
        [data-testid="stSidebar"] p {
            color: #BAE6FD !important;
            line-height: 1.7 !important;
        }
        
        /* Input field tooltip icon - ENHANCED */
        .stTooltipIcon {
            color: #00C9FF !important;
            font-size: 22px !important;
        }
        
        /* Tooltip content styling - ENHANCED */
        .stMarkdown div[data-testid="stMarkdownContainer"] p {
            font-size: 15px !important;
            line-height: 1.7 !important;
            color: #1E293B !important;
        }
        
        /* Form group styling - ENHANCED */
        .form-group {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 14px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #00C9FF;
            border-top: 1px solid #E0F2FE;
            border-right: 1px solid #E0F2FE;
            border-bottom: 1px solid #E0F2FE;
        }
        
        .form-group:hover {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        
        /* Form group title - ENHANCED */
        .form-group-title {
            font-size: 20px;
            font-weight: 600;
            color: #0284C7;
            margin-bottom: 18px;
            border-bottom: 2px solid #E0F2FE;
            padding-bottom: 12px;
            text-align: left;
            letter-spacing: 0.3px;
        }
        
        /* Divider styling - ENHANCED */
        .custom-divider {
            height: 3px;
            background: linear-gradient(90deg, transparent, #00C9FF, #92FE9D, transparent);
            margin: 35px 0;
            opacity: 0.7;
            border-radius: 3px;
        }
        
        /* Risk factors and retention strengths boxes - ENHANCED */
        div[style*='background-color: rgba(255, 220, 220, 0.3)'] {
            background-color: rgba(254, 226, 226, 0.7) !important;
            border-radius: 12px !important;
            border-left: 5px solid #EF4444 !important;
            box-shadow: 0 5px 15px rgba(239, 68, 68, 0.15) !important;
        }
        
        div[style*='background-color: rgba(220, 255, 220, 0.3)'] {
            background-color: rgba(220, 252, 231, 0.7) !important;
            border-radius: 12px !important;
            border-left: 5px solid #10B981 !important;
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.15) !important;
        }
        
        /* Headers in result boxes - NEW */
        div[style*='background-color: rgba(254, 226, 226, 0.7)'] h4,
        div[style*='background-color: rgba(220, 252, 231, 0.7)'] h4 {
            font-size: 18px !important;
            font-weight: 600 !important;
            margin-bottom: 12px !important;
            color: #1E293B !important;
        }
        
        /* List items in result boxes - NEW */
        div[style*='background-color: rgba(254, 226, 226, 0.7)'] ul,
        div[style*='background-color: rgba(220, 252, 231, 0.7)'] ul {
            margin-left: 20px !important;
        }
        
        div[style*='background-color: rgba(254, 226, 226, 0.7)'] li,
        div[style*='background-color: rgba(220, 252, 231, 0.7)'] li {
            margin-bottom: 8px !important;
            color: #334155 !important;
            font-size: 15px !important;
        }
        
        /* Footer styling - ENHANCED */
        div[style*='text-align: center; margin-top: 40px'] {
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.8), rgba(30, 58, 138, 0.7), rgba(15, 23, 42, 0.8)) !important;
            padding: 25px !important;
            border-radius: 15px !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
        }
        
        div[style*='text-align: center; margin-top: 40px'] p {
            color: #E0F2FE !important;
            font-size: 1rem !important;
            letter-spacing: 0.5px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown("<h1 class='main-title'>🔍 Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center;'>Prediction Options</h2>", unsafe_allow_html=True)
    
    model_type = st.radio(
        "Choose the type of Churn Prediction:",
        ["Bank Customer", "Telecom Customer"],
        format_func=lambda x: f"📊 {x}"
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.1); padding: 18px; border-radius: 12px; margin-top: 25px; border: 1px solid rgba(0, 201, 255, 0.3);'>
        <h3 style='color: #BAE6FD; text-align: center; font-size: 1.5rem; margin-bottom: 15px;'>About</h3>
        <p style='color: #BAE6FD; font-size: 0.95rem; line-height: 1.6;'>
            This app uses machine learning models to predict customer churn in Bank and Telecom sectors.
            Enter customer details to predict whether they are likely to churn.
        </p>
        <p style='color: #BAE6FD; font-size: 0.95rem; text-align: center; font-style: italic; margin-top: 18px;'>
            Developed by Fathima Shabna Ilmi
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main content
if model_type == "Bank Customer":
    st.markdown("<div class='form-container'><h2 class='section-header'>🏦 Bank Customer Churn Prediction</h2>", unsafe_allow_html=True)
    
    with st.form(key="bank_form"):
        # Basic Customer Information Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>📋 Customer Profile</h3>", unsafe_allow_html=True)
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, help="Customer's age")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1, help="Customer's credit score (300-900)")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Account Information Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>💰 Account Information</h3>", unsafe_allow_html=True)
        
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, help="Years as a customer")
        balance = st.number_input("Balance", help="Current account balance")
        estimated_salary = st.number_input("Estimated Salary", help="Customer's estimated annual salary")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Product Usage Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>🛒 Product Usage</h3>", unsafe_allow_html=True)
        
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, help="Number of bank products used")
        has_cr_card = st.radio("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Engagement Metrics Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>📊 Engagement Metrics</h3>", unsafe_allow_html=True)
        
        is_active_member = st.radio("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, help="Customer satisfaction rating (1-5)")
        points_earned = st.number_input("Points Earned", min_value=0, help="Reward points earned")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Divider
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("📊 Predict Churn")
        with col2:
            clear_button = st.form_submit_button("🔄 Clear Form")

        if clear_button:
            st.experimental_rerun()

        if submit_button:
            if (credit_score == 0 or age == 0 or balance == 0 or estimated_salary == 0 or points_earned == 0):
                st.error("Please fill in all the fields correctly before submitting.")
            else:
                with st.spinner("Analyzing customer data..."):
                    # Add a slight delay for effect
                    import time
                    time.sleep(1)
                    
                    gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]
                    card_type_encoded = [1 if card_type == "DIAMOND" else 0, 1 if card_type == "GOLD" else 0, 1 if card_type == "SILVER" else 0, 1 if card_type == "PLATINUM" else 0]
                    features = np.array([credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, satisfaction_score, points_earned] + gender_encoded + card_type_encoded)
                    result = predict_churn(bank_model, bank_scaler, features)
                    
                    if result == "Churned":
                        st.error(f"⚠️ Prediction: This customer is likely to churn!")
                        st.markdown("""
                        <div style='background-color: rgba(255, 220, 220, 0.3); padding: 18px; border-radius: 12px; border-left: 5px solid #ff5252;'>
                            <h4>Risk Factors:</h4>
                            <ul>
                                <li>Consider reviewing their account benefits</li>
                                <li>Reach out to improve satisfaction</li>
                                <li>Offer personalized retention incentives</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ Prediction: This customer is likely to remain!")
                        st.markdown("""
                        <div style='background-color: rgba(220, 255, 220, 0.3); padding: 18px; border-radius: 12px; border-left: 5px solid #4CAF50;'>
                            <h4>Retention Strengths:</h4>
                            <ul>
                                <li>Consider upselling additional products</li>
                                <li>Encourage referrals from this loyal customer</li>
                                <li>Monitor for any changes in engagement patterns</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif model_type == "Telecom Customer":
    st.markdown("<div class='form-container'><h2 class='section-header'>📞 Telecom Customer Churn Prediction</h2>", unsafe_allow_html=True)
    
    with st.form(key="telecom_form"):
        # Customer Profile Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>👤 Customer Demographics</h3>", unsafe_allow_html=True)
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, help="How long the customer has been with the company")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Billing Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>💵 Billing Information</h3>", unsafe_allow_html=True)
        
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, help="Monthly bill amount")
        total_charges = st.number_input("Total Charges", min_value=0.0, help="Total amount charged to date")
        paperless_billing = st.radio("Paperless Billing?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Service Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>🌐 Service Details</h3>", unsafe_allow_html=True)
        
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], help="Contract length")
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], help="Type of internet service")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Divider
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("📊 Predict Churn")
        with col2:
            clear_button = st.form_submit_button("🔄 Clear Form")

        if clear_button:
            st.experimental_rerun()

        if submit_button:
            if (monthly_charges == 0 or total_charges == 0 or tenure == 0):
                st.error("Please fill in all the fields correctly before submitting.")
            else:
                with st.spinner("Analyzing telecom data..."):
                    # Add a slight delay for effect
                    import time
                    time.sleep(1)
                    
                    contract_encoded = [1 if contract == "Month-to-month" else 0, 1 if contract == "One year" else 0, 1 if contract == "Two year" else 0]
                    internet_service_encoded = [1 if internet_service == "Fiber optic" else 0, 1 if internet_service == "DSL" else 0, 1 if internet_service == "No" else 0]
                    payment_method_encoded = [1 if payment_method == "Electronic check" else 0, 1 if payment_method == "Mailed check" else 0, 1 if payment_method == "Bank transfer (automatic)" else 0, 1 if payment_method == "Credit card (automatic)" else 0]
                    gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]
                    features = np.array([paperless_billing, monthly_charges, total_charges, tenure] + contract_encoded + internet_service_encoded + payment_method_encoded + gender_encoded)
                    result = predict_churn(telecom_model, telecom_scaler, features)
                    
                    if result == "Churned":
                        st.error(f"⚠️ Prediction: This customer is likely to churn!")
                        st.markdown("""
                        <div style='background-color: rgba(255, 220, 220, 0.3); padding: 18px; border-radius: 12px; border-left: 5px solid #ff5252;'>
                            <h4>Risk Factors:</h4>
                            <ul>
                                <li>Review contract terms and offer upgrades</li>
                                <li>Consider service quality improvements</li>
                                <li>Provide competitive pricing options</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ Prediction: This customer is likely to remain!")
                        st.markdown("""
                        <div style='background-color: rgba(220, 255, 220, 0.3); padding: 18px; border-radius: 12px; border-left: 5px solid #4CAF50;'>
                            <h4>Retention Strengths:</h4>
                            <ul>
                                <li>Consider offering loyalty rewards</li>
                                <li>Opportunity for service upgrades</li>
                                <li>Monitor for competitive offers they may receive</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Video controls - Optional feature
with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #BAE6FD; text-align: center; margin-bottom: 15px;'>Video Controls</h3>", unsafe_allow_html=True)
    
    video_opacity = st.slider("Background Opacity", 0.1, 1.0, 0.6, 0.1)
    
    # Apply opacity change with JavaScript
    st.markdown(
        f"""
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {{
                const videoElem = document.querySelector('.video-background');
                if (videoElem) {{
                    videoElem.style.opacity = "{video_opacity}";
                }}
            }});
        </script>
        """,
        unsafe_allow_html=True
    )


# Footer with animated gradient
st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 20px; background: linear-gradient(90deg, rgba(0,0,0,0.7), rgba(0,0,0,0.5), rgba(0,0,0,0.7)); border-radius: 10px; animation: gradientBG 10s ease infinite;'>
    <p style='color: #ddd; font-size: 0.9rem;'>
        © 2025 Churn Prediction Tool | Built with Streamlit
    </p>
</div>
<style>
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)