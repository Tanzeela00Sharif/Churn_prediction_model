import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ---------------------------------------------------------
page_icon = Image.open("house_icon.png") if os.path.exists("house_icon.png") else "🏠"
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM STYLING (theme-aware — no hardcoded light backgrounds)
# ---------------------------------------------------------
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #4F46E5;
            color: white;
            font-weight: 600;
            padding: 0.6rem;
            border-radius: 8px;
            border: none;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #4338CA;
            color: white;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1rem;
            border: 1px solid rgba(250,250,250,0.15);
            background-color: rgba(250,250,250,0.04);
        }
        .churn-risk {
            border-left: 4px solid #EF4444;
        }
        .no-churn-risk {
            border-left: 4px solid #22C55E;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL, SCALER, COLUMNS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_churn.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure logistic_churn.pkl, scaler.pkl, "
        "and model_columns.pkl are in the same folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("📊 Customer Churn Predictor")
st.markdown(
    "Predict whether a customer is likely to churn based on their profile "
    "and account details. Fill in the customer information in the sidebar, "
    "then click **Predict**."
)

with st.expander("ℹ️ About this model"):
    st.markdown(
        "This app uses a **Logistic Regression** model trained on customer "
        "account and service data to estimate churn probability. Adjust the "
        "decision threshold to control sensitivity — lower values flag more "
        "customers as at-risk (more false alarms), higher values are stricter."
    )

st.divider()

# ---------------------------------------------------------
# SIDEBAR — INPUT FORM
# ---------------------------------------------------------
st.sidebar.header("Customer Details")

def user_input_form():
    c1, c2 = st.sidebar.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
    with c2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.sidebar.markdown("---")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    st.sidebar.markdown("---")
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    st.sidebar.markdown("---")
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0)

    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([data])

input_df = user_input_form()

# ---------------------------------------------------------
# PREPROCESS INPUT TO MATCH TRAINING FORMAT
# ---------------------------------------------------------
def preprocess(input_df, model_columns):
    df_encoded = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_columns]
    return df_encoded

# ---------------------------------------------------------
# MAIN PANEL — SUMMARY + PREDICTION
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Summary")
    with st.container(border=True):
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Tenure", f"{input_df['tenure'].values[0]} months")
            st.metric("Monthly Charges", f"${input_df['MonthlyCharges'].values[0]:.2f}")
        with m2:
            st.metric("Contract", input_df['Contract'].values[0])
            st.metric("Total Charges", f"${input_df['TotalCharges'].values[0]:.2f}")
        st.caption(f"🌐 Internet Service: **{input_df['InternetService'].values[0]}**")

with col2:
    st.subheader("Prediction")
    threshold = st.slider(
        "Decision threshold",
        0.10, 0.90, 0.35, 0.05,
        help="Lower threshold = catches more potential churners, but more false alarms."
    )

    if st.button("🔍 Predict Churn Risk"):
        processed_input = preprocess(input_df, model_columns)
        scaled_input = scaler.transform(processed_input)

        probability = model.predict_proba(scaled_input)[0][1]
        prediction = 1 if probability >= threshold else 0

        # --- Gauge chart ---
        gauge_color = "#EF4444" if prediction == 1 else "#22C55E"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, threshold * 100], 'color': "rgba(34,197,94,0.15)"},
                    {'range': [threshold * 100, 100], 'color': "rgba(239,68,68,0.15)"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.8,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=250,
            margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Result card ---
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card churn-risk">
                <h3>⚠️ High Churn Risk</h3>
                <p style="font-size:1.4rem; font-weight:700; color:#F87171;">
                    {probability*100:.1f}% probability
                </p>
                <p>This customer is likely to churn. Consider a retention offer.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card no-churn-risk">
                <h3>✅ Low Churn Risk</h3>
                <p style="font-size:1.4rem; font-weight:700; color:#4ADE80;">
                    {probability*100:.1f}% probability
                </p>
                <p>This customer is likely to stay.</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Built by <b>Tanzila Sharif</b> · "
    "<a href='https://linkedin.com/in/tanzilasharif' target='_blank'>LinkedIn</a>"
    "</p>",
    unsafe_allow_html=True
)
