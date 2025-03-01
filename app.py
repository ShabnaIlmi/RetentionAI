import streamlit as st
import numpy as np
import joblib
from PIL import Image

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
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
        }
        h1 {
            color: #2E3B55;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Churn Prediction App")
st.markdown("Use this app to predict customer churn for Bank and Telecom sectors.")

# Sidebar for Image Upload
st.sidebar.title("Upload Customer Picture")
uploaded_image = st.sidebar.file_uploader("Choose a picture...", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.sidebar.image(image, caption="Uploaded Customer Picture", use_column_width=True)

# Model Selection
model_type = st.radio("Choose the type of Churn Prediction:", ["Bank Customer", "Telecom Customer"])

if model_type == "Bank Customer":
    st.header("🏦 Bank Customer Churn Prediction")

    # Input fields with custom style
    with st.form(key="bank_form"):
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10)
        balance = st.number_input("Balance")
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
        has_cr_card = st.radio("Has Credit Card?", [0, 1])
        is_active_member = st.radio("Is Active Member?", [0, 1])
        estimated_salary = st.number_input("Estimated Salary")
        satisfaction_score = st.slider("Satisfaction Score", 1, 5)
        card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
        points_earned = st.number_input("Points Earned", min_value=0)
        submit_button = st.form_submit_button("Predict")

        if submit_button:
            if (credit_score == 0 or age == 0 or balance == 0 or estimated_salary == 0 or points_earned == 0):
                st.error("Please fill in all the fields correctly before submitting.")
            else:
                gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]
                card_type_encoded = [1 if card_type == "DIAMOND" else 0, 1 if card_type == "GOLD" else 0, 1 if card_type == "SILVER" else 0, 1 if card_type == "PLATINUM" else 0]
                features = np.array([credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, satisfaction_score, points_earned] + gender_encoded + card_type_encoded)
                result = predict_churn(bank_model, bank_scaler, features)
                st.success(f"Predicted Churn Status: {result}")

elif model_type == "Telecom Customer":
    st.header("📞 Telecom Customer Churn Prediction")
    with st.form(key="telecom_form"):
        tenure = st.number_input("Tenure", min_value=0, max_value=100)
        monthly_charges = st.number_input("Monthly Charges")
        total_charges = st.number_input("Total Charges")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = st.radio("Paperless Billing?", [0, 1])
        gender = st.selectbox("Gender", ["Male", "Female"])
        submit_button = st.form_submit_button("Predict")

        if submit_button:
            if (monthly_charges == 0 or total_charges == 0 or tenure == 0):
                st.error("Please fill in all the fields correctly before submitting.")
            else:
                contract_encoded = [1 if contract == "Month-to-month" else 0, 1 if contract == "One year" else 0, 1 if contract == "Two year" else 0]
                internet_service_encoded = [1 if internet_service == "Fiber optic" else 0, 1 if internet_service == "DSL" else 0, 1 if internet_service == "No" else 0]
                payment_method_encoded = [1 if payment_method == "Electronic check" else 0, 1 if payment_method == "Mailed check" else 0, 1 if payment_method == "Bank transfer (automatic)" else 0, 1 if payment_method == "Credit card (automatic)" else 0]
                gender_encoded = [1 if gender == "Male" else 0, 1 if gender == "Female" else 0]
                features = np.array([paperless_billing, monthly_charges, total_charges, tenure] + contract_encoded + internet_service_encoded + payment_method_encoded + gender_encoded)
                result = predict_churn(telecom_model, telecom_scaler, features)
                st.success(f"Predicted Churn Status: {result}")

st.sidebar.markdown("""
### About
This app uses machine learning models to predict customer churn in Bank and Telecom sectors.
Developed by Fathima Shabna Ilmi
""")
