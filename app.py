import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
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
        }
        .churn-risk {
            background-color: #FEF2F2;
            border: 1px solid #FCA5A5;
        }
        .no-churn-risk {
            background-color: #F0FDF4;
            border: 1px solid #86EFAC;
        }
        .metric-box {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
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
        "Model files not found. Make sure churn_model.pkl, scaler.pkl, "
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
st.divider()

# ---------------------------------------------------------
# SIDEBAR — INPUT FORM
# ---------------------------------------------------------
st.sidebar.header("Customer Details")

def user_input_form():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

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
    # Add any missing columns the model expects, fill with 0
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    # Keep only the columns the model was trained on, in the right order
    df_encoded = df_encoded[model_columns]
    return df_encoded

# ---------------------------------------------------------
# MAIN PANEL — SUMMARY + PREDICTION
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Summary")
    st.markdown(f"""
    <div class="metric-box">
        <b>Tenure:</b> {input_df['tenure'].values[0]} months<br>
        <b>Contract:</b> {input_df['Contract'].values[0]}<br>
        <b>Monthly Charges:</b> ${input_df['MonthlyCharges'].values[0]:.2f}<br>
        <b>Total Charges:</b> ${input_df['TotalCharges'].values[0]:.2f}<br>
        <b>Internet Service:</b> {input_df['InternetService'].values[0]}
    </div>
    """, unsafe_allow_html=True)

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

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card churn-risk">
                <h3>⚠️ High Churn Risk</h3>
                <p style="font-size:1.5rem; font-weight:700; color:#DC2626;">
                    {probability*100:.1f}% probability
                </p>
                <p>This customer is likely to churn. Consider a retention offer.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card no-churn-risk">
                <h3>✅ Low Churn Risk</h3>
                <p style="font-size:1.5rem; font-weight:700; color:#16A34A;">
                    {probability*100:.1f}% probability
                </p>
                <p>This customer is likely to stay.</p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(int(probability * 100), 100))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
    "Built by <b>Tanzila Sharif</b> · "
    "<a href='https://linkedin.com/in/tanzilasharif' target='_blank'>LinkedIn</a> · "
    "</p>",
    unsafe_allow_html=True
)
