import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Churn Prediction",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a3c5e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .result-card {
        background: #f8faff;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #1a3c5e;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .exits { border-left-color: #e74c3c !important; }
    .stays { border-left-color: #27ae60 !important; }
    .badge-exits {
        background: #fdecea;
        color: #c0392b;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
    }
    .badge-stays {
        background: #eafaf1;
        color: #1e8449;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a3c5e;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #d0e4f7;
        padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Decision Tree":        joblib.load("models/nate_decision_tree.sav"),
        "K-Nearest Neighbors":  joblib.load("models/nate_knn.sav"),
        "Logistic Regression":  joblib.load("models/nate_logistic_regression.sav"),
        "Random Forest":        joblib.load("models/nate_random_forest.sav"),
        "SVM":                  joblib.load("models/SVM_model.sav"),
        "XGBoost":              joblib.load("models/XGBoost_model.sav"),
    }

try:
    loaded_models = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Helper ─────────────────────────────────────────────────────────────────────
def decode(pred):
    return "Customer Exits" if pred == 1 else "Customer Stays"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Banking Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter customer details below and run all six ML models at once.</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load model files.\n\n{load_error}")
    st.stop()

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("churn_form"):
    st.markdown('<div class="section-header">Customer Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score   = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        geography      = st.selectbox("Geography", options=[0, 1, 2],
                                     format_func=lambda x: {0: "France", 1: "Germany", 2: "Spain"}[x])
        gender         = st.selectbox("Gender", options=[0, 1],
                                     format_func=lambda x: {0: "Female", 1: "Male"}[x])

    with col2:
        age            = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure         = st.number_input("Tenure", min_value=0, max_value=10, value=5)
        balance        = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0)

    with col3:
        num_products   = st.number_input("Products", min_value=1, max_value=4, value=1)
        has_cr_card    = st.selectbox("Credit Card", options=[1, 0],
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        is_active      = st.selectbox("Active Member", options=[1, 0],
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        salary         = st.number_input("Salary", min_value=0.0, max_value=300000.0, value=60000.0)

    submitted = st.form_submit_button("Predict")

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    geo_map    = {0: "France", 1: "Germany", 2: "Spain"}
    gender_map = {0: "Female", 1: "Male"}

    input_dict = {
        'CreditScore': int(credit_score),
        'Geography': geo_map[geography],
        'Gender': gender_map[gender],
        'Age': int(age),
        'Tenure': int(tenure),
        'Balance': float(balance),
        'NumOfProducts': int(num_products),
        'HasCrCard': int(has_cr_card),
        'IsActiveMember': int(is_active),
        'EstimatedSalary': float(salary)
    }

    input_df = pd.DataFrame([input_dict])

    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

    for name, model in loaded_models.items():
        try:
            pred = model.predict(input_df)[0]
            st.write(f"{name}: {decode(pred)}")
        except Exception as e:
            st.error(f"{name} failed: {e}")
