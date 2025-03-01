import streamlit as st
import numpy as np
import joblib
from PIL import Image
import base64

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

# Function to get base64 encoded image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI
st.set_page_config(page_title="Churn Prediction App", page_icon="🔍", layout="wide")

# Set background image
set_bg_from_url("https://source.unsplash.com/1600x900/?digital,blue")

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
            color: #ffffff;
            font-size: 3.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            margin: 1.5rem 0;
            background: linear-gradient(90deg, #3a7bd5, #00d2ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 15px;
        }
        
        /* Form container styling */
        .form-container {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            margin: 20px 0;
            border-left: 5px solid #3a7bd5;
        }
        
        /* Section headers */
        .section-header {
            color: #3a7bd5;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 2px solid #3a7bd5;
            padding-bottom: 10px;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input {
            border-radius: 10px;
            padding: 12px;
            border: 2px solid #ddd;
            transition: all 0.3s ease;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .stTextInput>div>div>input:focus, 
        .stNumberInput>div>div>input:focus {
            border-color: #3a7bd5;
            box-shadow: 0 0 0 2px rgba(58, 123, 213, 0.2);
        }
        
        /* Select box styling */
        .stSelectbox>div>div {
            border-radius: 10px;
            border: 2px solid #ddd;
        }
        
        .stSelectbox>div>div:focus-within {
            border-color: #3a7bd5;
            box-shadow: 0 0 0 2px rgba(58, 123, 213, 0.2);
        }
        
        /* Radio button styling */
        .stRadio>div {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Slider styling */
        .stSlider>div>div>div>div {
            background-color: #3a7bd5;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #3a7bd5, #00d2ff);
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 12px 24px;
            border: none;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
        }
        
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        /* Success message styling */
        .element-container div[data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            animation: fadeIn 0.5s ease-in-out;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: rgba(20, 40, 80, 0.85);
        }
        
        .css-1d391kg .block-container {
            padding: 2rem 1rem;
        }
        
        /* Two-column layout for form fields */
        .form-row {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            gap: 20px;
        }
        
        .form-col {
            flex: 1;
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
    <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 20px;'>
        <h3 style='color: white; text-align: center;'>About</h3>
        <p style='color: white; font-size: 0.9rem;'>
            This app uses machine learning models to predict customer churn in Bank and Telecom sectors.
            Enter customer details to predict whether they are likely to churn.
        </p>
        <p style='color: white; font-size: 0.9rem; text-align: center; font-style: italic; margin-top: 15px;'>
            Developed by Fathima Shabna Ilmi
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main content
if model_type == "Bank Customer":
    st.markdown("<div class='form-container'><h2 class='section-header'>🏦 Bank Customer Churn Prediction</h2>", unsafe_allow_html=True)
    
    with st.form(key="bank_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1, help="Customer's credit score (300-900)")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, help="Customer's age")
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, help="Years as a customer")
            balance = st.number_input("Balance", help="Current account balance")
        
        with col2:
            num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, help="Number of bank products used")
            has_cr_card = st.radio("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active_member = st.radio("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input("Estimated Salary", help="Customer's estimated annual salary")
            satisfaction_score = st.slider("Satisfaction Score", 1, 5, help="Customer satisfaction rating (1-5)")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        
        with col4:
            points_earned = st.number_input("Points Earned", min_value=0, help="Reward points earned")
        
        col5, col6 = st.columns(2)
        
        with col5:
            submit_button = st.form_submit_button("📊 Predict Churn")
        
        with col6:
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
                        <div style='background-color: rgba(255, 220, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #ff5252;'>
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
                        <div style='background-color: rgba(220, 255, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
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
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, help="How long the customer has been with the company")
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, help="Monthly bill amount")
            total_charges = st.number_input("Total Charges", min_value=0.0, help="Total amount charged to date")
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], help="Contract length")
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], help="Type of internet service")
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless_billing = st.radio("Paperless Billing?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        col3, col4 = st.columns(2)
        
        with col3:
            submit_button = st.form_submit_button("📊 Predict Churn")
        
        with col4:
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
                        <div style='background-color: rgba(255, 220, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #ff5252;'>
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
                        <div style='background-color: rgba(220, 255, 220, 0.3); padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                            <h4>Retention Strengths:</h4>
                            <ul>
                                <li>Consider offering loyalty rewards</li>
                                <li>Opportunity for service upgrades</li>
                                <li>Monitor for competitive offers they may receive</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 20px; background-color: rgba(0,0,0,0.5); border-radius: 10px;'>
    <p style='color: #ddd; font-size: 0.8rem;'>
        © 2025 Churn Prediction Tool | Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
