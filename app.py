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
            background: linear-gradient(135deg, #121638, #2E3192);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom CSS for advanced form styling with enhanced colors
st.markdown(
    """
    <style>
        /* Main app styling with modern color palette */
        .stApp {
            font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Enhanced Main title styling with vibrant gradient */
        .main-title {
            text-align: center;
            font-size: 3.8rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            margin: 2rem 0;
            background: linear-gradient(90deg, #6A11CB, #2575FC, #49C6E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 25px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 18px;
            animation: glow 3s ease-in-out infinite alternate;
            letter-spacing: 1px;
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 5px #fff, 0 0 10px #6A11CB, 0 0 15px #2575FC;
            }
            to {
                text-shadow: 0 0 10px #fff, 0 0 20px #2575FC, 0 0 30px #49C6E5;
            }
        }
        
        /* Form container styling with modern glassmorphism effect */
        .form-container {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 24px;
            padding: 35px;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
            margin: 25px 0;
            border-left: 6px solid #6A11CB;
            border-top: 1px solid rgba(255, 255, 255, 0.4);
            border-bottom: 1px solid rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(10px);
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        
        .form-container:hover {
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.5);
            transform: translateY(-7px);
        }
        
        /* Section headers with improved gradient */
        .section-header {
            color: #2575FC;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 30px;
            border-bottom: 3px solid transparent;
            border-image: linear-gradient(to right, #6A11CB, #2575FC, #49C6E5);
            border-image-slice: 1;
            padding-bottom: 15px;
            text-align: center;
            letter-spacing: 0.5px;
        }
        
        /* Input field styling with enhanced focus state */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input {
            border-radius: 12px;
            padding: 14px 18px;
            border: 2px solid #E6E6FA;
            transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 15px;
            font-size: 16px;
            width: 100%;
            background-color: rgba(255, 255, 255, 0.9);
        }
        
        .stTextInput>div>div>input:focus, 
        .stNumberInput>div>div>input:focus {
            border-color: #6A11CB;
            box-shadow: 0 0 0 4px rgba(106, 17, 203, 0.2);
            transform: translateY(-3px);
            background-color: #ffffff;
        }
        
        /* Label styling with modern color */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .css-16huue1 {
            font-weight: 600 !important;
            font-size: 17px !important;
            color: #121638 !important;
            margin-bottom: 10px !important;
            letter-spacing: 0.3px !important;
        }
        
        /* Input field container with animated border */
        .input-container {
            margin-bottom: 28px;
            border-bottom: 1px dashed #E6E6FA;
            padding-bottom: 18px;
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        
        .input-container:hover {
            border-bottom-color: #6A11CB;
        }
        
        .input-container:hover::after {
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, #6A11CB, #2575FC, #49C6E5);
            animation: slideIn 0.5s forwards;
        }
        
        @keyframes slideIn {
            from { transform: translateX(-100%); }
            to { transform: translateX(0); }
        }
        
        /* Select box styling with improved focus state */
        .stSelectbox>div>div {
            border-radius: 12px !important;
            border: 2px solid #E6E6FA !important;
            margin-bottom: 15px;
            background-color: rgba(255, 255, 255, 0.9) !important;
            transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        
        .stSelectbox>div>div:focus-within {
            border-color: #6A11CB !important;
            box-shadow: 0 0 0 4px rgba(106, 17, 203, 0.2) !important;
            transform: translateY(-3px);
            background-color: #ffffff !important;
        }
        
        /* Radio button styling with updated colors */
        .stRadio>div {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
            margin-bottom: 15px;
            border: 1px solid rgba(230, 230, 250, 0.8);
            transition: all 0.3s ease;
        }
        
        .stRadio>div:hover {
            border-color: #6A11CB;
            background-color: rgba(255, 255, 255, 0.95);
            transform: translateY(-2px);
            box-shadow: 0 5px 12px rgba(0,0,0,0.12);
        }
        
        /* Slider styling with updated colors */
        .stSlider>div>div>div>div {
            background-color: #2575FC !important;
        }
        
        .stSlider>div>div>div {
            background: linear-gradient(to right, rgba(106, 17, 203, 0.1), rgba(37, 117, 252, 0.2)) !important;
        }
        
        /* Button styling with dynamic gradient */
        .stButton>button {
            background: linear-gradient(45deg, #6A11CB, #2575FC, #49C6E5);
            color: white;
            font-weight: 600;
            border-radius: 12px;
            padding: 14px 28px;
            border: none;
            box-shadow: 0 5px 18px rgba(0, 0, 0, 0.2);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-size: 16px;
            width: 100%;
            margin-top: 20px;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, #49C6E5, #2575FC, #6A11CB);
            transition: all 0.6s ease;
            z-index: -1;
        }
        
        .stButton>button:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            letter-spacing: 2px;
        }
        
        .stButton>button:hover::before {
            left: 0;
        }
        
        .stButton>button:active {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Success message styling with enhanced effects */
        .element-container div[data-testid="stAlert"] {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 16px;
            padding: 25px;
            margin: 25px 0;
            animation: fadeInUp 0.7s ease-in-out;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            border-left: 6px solid #2575FC;
        }
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Sidebar styling with improved gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(18, 22, 56, 0.97), rgba(106, 17, 203, 0.92)) !important;
            box-shadow: 3px 0 20px rgba(0, 0, 0, 0.35);
        }
        
        [data-testid="stSidebar"] .block-container {
            padding: 3rem 1.8rem;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            margin-bottom: 25px !important;
            background: linear-gradient(90deg, #ffffff, #E6E6FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 26px !important;
        }
        
        /* Input field tooltip icon */
        .stTooltipIcon {
            color: #2575FC !important;
            font-size: 22px !important;
            transition: all 0.3s ease;
        }
        
        .stTooltipIcon:hover {
            color: #6A11CB !important;
            transform: scale(1.2);
        }
        
        /* Form group styling with enhanced effects */
        .form-group {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 28px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #6A11CB;
            border-top: 1px solid rgba(255, 255, 255, 0.6);
            border-right: 1px solid rgba(255, 255, 255, 0.6);
            border-bottom: 1px solid rgba(255, 255, 255, 0.6);
            position: relative;
            overflow: hidden;
            transition: all 0.4s ease;
        }
        
        .form-group:hover {
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            transform: translateY(-4px);
            background-color: rgba(255, 255, 255, 0.92);
        }
        
        .form-group::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #6A11CB, #2575FC, #49C6E5);
            opacity: 0.8;
        }
        
        .form-group-title {
            font-size: 20px;
            font-weight: 700;
            color: #121638;
            margin-bottom: 18px;
            border-bottom: 1px solid #E6E6FA;
            padding-bottom: 12px;
            text-align: left;
            letter-spacing: 0.5px;
        }
        
        /* Enhanced divider styling */
        .custom-divider {
            height: 3px;
            background: linear-gradient(90deg, transparent, #6A11CB, #2575FC, #49C6E5, transparent);
            margin: 35px 0;
            opacity: 0.8;
            border-radius: 3px;
        }
        
        /* Risk and Success messages with enhanced styling */
        .risk-factors {
            background: linear-gradient(to right, rgba(255, 0, 0, 0.03), rgba(255, 100, 100, 0.07));
            padding: 20px;
            border-radius: 14px;
            border-left: 6px solid #FF5252;
            box-shadow: 0 4px 12px rgba(255, 82, 82, 0.1);
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        
        .risk-factors:hover {
            box-shadow: 0 6px 18px rgba(255, 82, 82, 0.15);
            transform: translateY(-3px);
        }
        
        .retention-strengths {
            background: linear-gradient(to right, rgba(76, 175, 80, 0.03), rgba(76, 175, 80, 0.07));
            padding: 20px;
            border-radius: 14px;
            border-left: 6px solid #4CAF50;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.1);
            margin-top: 20px;
            transition: all 0.3s ease;
        }
        
        .retention-strengths:hover {
            box-shadow: 0 6px 18px rgba(76, 175, 80, 0.15);
            transform: translateY(-3px);
        }
        
        /* Enhanced footer styling */
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            background: linear-gradient(90deg, rgba(18, 22, 56, 0.8), rgba(106, 17, 203, 0.6), rgba(73, 198, 229, 0.8));
            background-size: 200% auto;
            border-radius: 16px;
            animation: gradientBG 15s ease infinite;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        
        .footer p {
            color: white;
            font-weight: 500;
            letter-spacing: 0.5px;
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Checkbox styling */
        .stCheckbox > div {
            padding: 15px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 12px;
            border: 1px solid #E6E6FA;
            transition: all 0.3s ease;
        }
        
        .stCheckbox > div:hover {
            background-color: rgba(255, 255, 255, 0.95);
            border-color: #2575FC;
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        }
        
        /* Notification and announcement styles */
        .notification {
            background: linear-gradient(to right, rgba(255, 158, 0, 0.05), rgba(255, 193, 7, 0.1));
            border-left: 5px solid #FFC107;
            padding: 18px;
            border-radius: 12px;
            margin: 20px 0;
            position: relative;
            box-shadow: 0 4px 15px rgba(255, 193, 7, 0.1);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(255, 193, 7, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0); }
        }
        
        /* Metrics and stats containers */
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 25px 0;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            flex: 1 1 200px;
            text-align: center;
            transition: all 0.3s ease;
            border-bottom: 3px solid #2575FC;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            border-bottom-color: #6A11CB;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #121638;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 14px;
            color: #666;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        /* Progress bar styling */
        .progress-container {
            width: 100%;
            height: 12px;
            background-color: rgba(230, 230, 250, 0.5);
            border-radius: 10px;
            margin: 15px 0;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(to right, #6A11CB, #2575FC);
            border-radius: 10px;
            transition: width 0.5s ease-in-out;
        }
        
        /* Enhanced expander styling */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            color: #121638 !important;
            background-color: rgba(230, 230, 250, 0.3) !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: rgba(106, 17, 203, 0.1) !important;
            color: #6A11CB !important;
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
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 14px; margin-top: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3 style='color: white; text-align: center; font-weight: 600; margin-bottom: 15px;'>About</h3>
        <p style='color: white; font-size: 0.95rem; line-height: 1.6;'>
            This app uses machine learning models to predict customer churn in Bank and Telecom sectors.
            Enter customer details to predict whether they are likely to churn.
        </p>
        <p style='color: white; font-size: 0.95rem; text-align: center; font-style: italic; margin-top: 15px; opacity: 0.9;'>
            Built with ❤️ using Streamlit and ML
        </p>
    </div>
    """, unsafe_allow_html=True)