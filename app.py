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
        
        /* Main title styling */
        .main-title {
            text-align: center;
            color: #003366;
            font-size: 3.5rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            margin: 1.5rem 0;
            background: linear-gradient(90deg, #FF5F6D, #FFC371);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 15px;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 5px #fff, 0 0 10px #ff9e00, 0 0 15px #ff9e00;
            }
            to {
                text-shadow: 0 0 10px #fff, 0 0 20px #ff9e00, 0 0 30px #ff9e00;
            }
        }
        
        /* Form container styling - ENHANCED with higher opacity for better contrast against video */
        .form-container {
            background-color: rgba(20, 20, 20, 0.9);  /* Darker background with slight opacity */
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6); /* Slightly stronger shadow for more contrast */
            margin: 20px 0;
            border-left: 5px solid #FF9E00; /* Orange highlight on the left */
            transition: all 0.3s ease;
        }

        .form-container:hover {
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.7); /* Stronger shadow on hover */
            transform: translateY(-5px); /* Lift effect on hover */
        }
        
        /* Section headers */
        .section-header {
            color: #FF9E00;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 25px;
            border-bottom: 2px solid #FF9E00;
            padding-bottom: 15px;
            text-align: center;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input {
            border-radius: 10px;
            padding: 12px 15px;
            border: 2px solid #ddd;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 12px;
            font-size: 16px;
            width: 100%;
        }
        
        .stTextInput>div>div>input:focus, 
        .stNumberInput>div>div>input:focus {
            border-color: #FF9E00;
            box-shadow: 0 0 0 3px rgba(255, 158, 0, 0.2);
            transform: translateY(-2px);
        }
        
        /* Label styling for ALL form elements - UPDATED to BLACK and increased specificity */
        .stTextInput > div > label, 
        .stNumberInput > div > label, 
        .stSelectbox > div > label, 
        .stRadio > div > label, 
        .stSlider > div > label, 
        .css-16huue1,
        .element-container p,
        p {
            font-weight: 600 !important;
            font-size: 17px !important;
            color: #000000 !important; /* Changed to black */
            margin-bottom: 8px !important;
            background-color: rgba(255, 255, 255, 0.9) !important; /* Increased opacity for better contrast */
            padding: 5px 10px !important;
            border-radius: 5px !important;
            display: inline-block !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* Make sure radio labels have the same styling */
        .stRadio > div > div > div > label {
            font-weight: 600 !important;
            font-size: 17px !important;
            color: #000000 !important;
            margin-bottom: 8px !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            padding: 5px 10px !important;
            border-radius: 5px !important;
            display: inline-block !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* Ensure all form labels have black text */
        .form-group-title, 
        .form-group label,
        [data-baseweb="radio"] span,
        [data-baseweb="select"] span,
        [data-baseweb="checkbox"] span {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Input field container */
        .input-container {
            margin-bottom: 25px;
            border-bottom: 1px dashed #eee;
            padding-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .input-container:hover {
            border-bottom-color: #FF9E00;
        }
        
        /* Select box styling */
        .stSelectbox>div>div {
            border-radius: 10px !important;
            border: 2px solid #ddd !important;
            margin-bottom: 12px;
        }
        
        .stSelectbox>div>div:focus-within {
            border-color: #FF9E00 !important;
            box-shadow: 0 0 0 3px rgba(255, 158, 0, 0.2) !important;
        }
        
        /* Radio button styling and text */
        .stRadio>div {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 12px;
            border: 1px solid #eee;
        }
        
        .stRadio>div:hover {
            border-color: #FF9E00;
        }
        
        /* Make radio button text black */
        .stRadio [data-testid="stMarkdownContainer"] p {
            color: #000000 !important;
            font-weight: 500 !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
        
        /* Style radio buttons labels */
        .stRadio label {
            background-color: rgba(255, 255, 255, 0.9) !important;
            padding: 5px 10px !important;
            border-radius: 5px !important;
            margin-bottom: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            display: inline-block !important;
        }
        
        /* Slider styling */
        .stSlider>div>div>div>div {
            background-color: #FF9E00 !important;
        }
        
        /* Button styling - ENHANCED */
        .stButton>button {
            background: linear-gradient(90deg, #FF5F6D, #FFC371);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 12px 24px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 16px;
            width: 100%;
            margin-top: 15px;
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button:before {
            content: "";
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: all 0.6s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 20px rgba(0, 0, 0, 0.3);
            background: linear-gradient(90deg, #FF5F6D, #FFC371);
        }
        
        .stButton>button:hover:before {
            left: 100%;
        }
        
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }
        
        /* Predict button specific styling */
        .stButton:nth-of-type(1)>button {
            background: linear-gradient(90deg, #36D1DC, #5B86E5);
            border-bottom: 4px solid #3672dc;
        }
        
        .stButton:nth-of-type(1)>button:hover {
            background: linear-gradient(90deg, #5B86E5, #36D1DC);
        }
        
        /* Clear button specific styling */
        .stButton:nth-of-type(2)>button {
            background: linear-gradient(90deg, #FF9966, #FF5E62);
            border-bottom: 4px solid #FF5E62;
        }
        
        .stButton:nth-of-type(2)>button:hover {
            background: linear-gradient(90deg, #FF5E62, #FF9966);
        }
        
        /* Success message styling */
        .element-container div[data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            animation: fadeIn 0.6s ease-in-out;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Sidebar styling - ENHANCED */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(20, 40, 80, 0.95), rgba(30, 60, 100, 0.9)) !important;
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.3);
        }
        
        [data-testid="stSidebar"] .block-container {
            padding: 2.5rem 1.5rem;
        }
        
        /* Input field tooltip icon */
        .stTooltipIcon {
            color: #FF9E00 !important;
            font-size: 20px !important;
        }
        
        /* Tooltip content styling */
        .stMarkdown div[data-testid="stMarkdownContainer"] p {
            font-size: 15px !important;
            line-height: 1.6 !important;
            color: #000000 !important; /* Ensure tooltip text is also black */
        }
        
        /* Form group styling */
        .form-group {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
            border-left: 4px solid #FF9E00;
        }

        /* Form group title styling - Ensure it's black */
        .form-group-title {
            font-size: 18px;
            font-weight: 600;
            color: #000000 !important; /* Explicitly black */
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            background-color: transparent !important; /* Remove any background */
            box-shadow: none !important; /* Remove any shadow */
            display: block !important; /* Full width */
        }
        
        /* Divider styling */
        .custom-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #FF9E00, transparent);
            margin: 30px 0;
            opacity: 0.7;
        }
        
        /* Force black text for all selectbox options */
        [data-baseweb="select-dropdown"] li,
        [data-baseweb="select-dropdown"] span {
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables if not already present
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# Fixed reset_form function that properly clears all form inputs
def reset_form():
    # Save the model_type to restore it after clearing
    current_model_type = st.session_state.get('model_type', 'Bank Customer')
    
    # List of keys to keep (don't reset)
    keys_to_keep = ['model_type']
    
    # Create a list of keys to remove (all except those in keys_to_keep)
    keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
    
    # Remove only the keys that should be removed
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset form_submitted flag
    st.session_state.form_submitted = False
    
    # Force rerun to show cleared form
    st.rerun()

# App Title
st.markdown("<h1 class='main-title'>🔍 Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    h2, h3, div.stRadio > div[role="radiogroup"] label, p {
        font-family: 'Poppins', sans-serif !important;
    }

    div.stRadio > div[role="radiogroup"] label {
        color: white !important;
        font-size: 1rem;
        font-weight: 400;
    }

    h2, h3 {
        font-weight: 600;
    }

    p {
        font-size: 0.9rem;
        font-weight: 300;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color: white; text-align: center;'>Prediction Options</h2>", unsafe_allow_html=True)

    # Store model_type in session_state
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "Bank Customer"
        
    model_type = st.radio(
        "Choose the type of Churn Prediction:",
        ["Bank Customer", "Telecom Customer"],
        key='model_type',
        format_func=lambda x: f"📊 {x}"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 20px;'>
        <h3 style='color: white; text-align: center;'>About</h3>
        <p style='color: white;'>
            This app uses machine learning models to predict customer churn in Bank and Telecom sectors.
            Enter customer details to predict whether they are likely to churn.
        </p>
        <p style='color: white; text-align: center; font-style: italic; margin-top: 15px;'>
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
        balance = st.number_input("Balance", min_value=0.0, help="Current account balance")
        estimated_salary = st.number_input("Estimated Salary", help="Customer's estimated annual salary")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Product Usage Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>🛒 Product Usage</h3>", unsafe_allow_html=True)
        
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, help="Number of bank products used")
        
        # Add a container div around the radio label for consistent styling
        st.markdown("<div class='radio-label-container'>", unsafe_allow_html=True)
        has_cr_card = st.radio("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        st.markdown("</div>", unsafe_allow_html=True)
        
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Engagement Metrics Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>📊 Engagement Metrics</h3>", unsafe_allow_html=True)
        
        # Add a container div around the radio label for consistent styling
        st.markdown("<div class='radio-label-container'>", unsafe_allow_html=True)
        is_active_member = st.radio("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        st.markdown("</div>", unsafe_allow_html=True)
        
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, help="Customer satisfaction rating (1-5)")
        points_earned = st.number_input("Points Earned", min_value=0, help="Reward points earned")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Divider
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Submit button inside form
        submit_button = st.form_submit_button("📊 Predict Churn")
            
        if submit_button:
            st.session_state.form_submitted = True
            if (credit_score == 0 or age == 0 or estimated_salary == 0 or points_earned == 0):
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
                    
                    # Store prediction result in session state
                    st.session_state.prediction_result = result
                    
                    if result == "Churned":
                        st.error(f"⚠️ Prediction: This customer is likely to churn!")
                        st.markdown("""
                        <div style='background-color: rgba(255, 220, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #ff5252;'>
                            <h4 style="color: #000000;">Risk Factors:</h4>
                            <ul style="color: #000000;">
                                <li>Consider reviewing their account benefits</li>
                                <li>Reach out to improve satisfaction</li>
                                <li>Offer personalized retention incentives</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ Prediction: This customer is likely to remain!")
                        st.markdown("""
                        <div style='background-color: rgba(220, 255, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                            <h4 style="color: #000000;">Retention Strengths:</h4>
                            <ul style="color: #000000;">
                                <li>Consider upselling additional products</li>
                                <li>Encourage referrals from this loyal customer</li>
                                <li>Monitor for any changes in engagement patterns</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    # Clear button outside the form
    if st.session_state.get('form_submitted', False):
        if st.button("🔄 Clear Form", key="bank_clear"):
            reset_form()

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
        
        # Add a container div around the radio label for consistent styling
        st.markdown("<div class='radio-label-container'>", unsafe_allow_html=True)
        paperless_billing = st.radio("Paperless Billing?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        st.markdown("</div>", unsafe_allow_html=True)
        
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Service Section
        st.markdown("<div class='form-group'><h3 class='form-group-title'>🌐 Service Details</h3>", unsafe_allow_html=True)
        
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], help="Contract length")
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], help="Type of internet service")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Divider
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # Submit button inside form
        submit_button = st.form_submit_button("📊 Predict Churn")

        if submit_button:
            st.session_state.form_submitted = True
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
                    
                    # Store prediction result in session state
                    st.session_state.prediction_result = result
                    
                    
                    if result == "Churned":
                        st.error(f"⚠️ Prediction: This customer is likely to churn!")
                        st.markdown("""
                        <div style='background-color: rgba(255, 220, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #ff5252;'>
                            <h4 style="color: #000000;">Risk Factors:</h4>
                            <ul style="color: #000000;">
                                <li>Consider reviewing their account benefits</li>
                                <li>Reach out to improve satisfaction</li>
                                <li>Offer personalized retention incentives</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success(f"✅ Prediction: This customer is likely to remain!")
                        st.markdown("""
                        <div style='background-color: rgba(220, 255, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                            <h4 style="color: #000000;">Retention Strengths:</h4>
                            <ul style="color: #000000;">
                                <li>Consider upselling additional products</li>
                                <li>Encourage referrals from this loyal customer</li>
                                <li>Monitor for any changes in engagement patterns</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    # Clear button outside the form
    if st.session_state.get('form_submitted', False):
        if st.button("🔄 Clear Form", key="bank_clear"):
            reset_form()

    st.markdown("</div>", unsafe_allow_html=True)

# Video controls - Optional feature
with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white; text-align: center;'>Video Controls</h3>", unsafe_allow_html=True)
    
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