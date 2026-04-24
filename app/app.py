import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load model and scaler ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return model, scaler, explainer

model, scaler, explainer = load_artifacts()

# ── Header ────────────────────────────────────────────────
st.title("🏦 Credit Risk Predictor")
st.markdown("Enter applicant details to predict the probability of loan default.")
st.divider()

# ── Sidebar — Input Form ──────────────────────────────────
st.sidebar.header("📋 Applicant Details")

revolving_utilization = st.sidebar.slider(
    "Revolving Utilization (0 = none, 1 = maxed out)",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01
)

age = st.sidebar.number_input(
    "Age", min_value=18, max_value=100, value=40
)

late_30_59 = st.sidebar.number_input(
    "Times 30-59 Days Late (past 2 years)",
    min_value=0, max_value=20, value=0
)

debt_ratio = st.sidebar.slider(
    "Debt Ratio (monthly debt / monthly income)",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01
)

monthly_income = st.sidebar.number_input(
    "Monthly Income ($)",
    min_value=0, max_value=100000, value=5000, step=500
)

open_credit_lines = st.sidebar.number_input(
    "Number of Open Credit Lines",
    min_value=0, max_value=50, value=5
)

late_90_days = st.sidebar.number_input(
    "Times 90+ Days Late",
    min_value=0, max_value=20, value=0
)

real_estate_loans = st.sidebar.number_input(
    "Number of Real Estate Loans",
    min_value=0, max_value=20, value=1
)

late_60_89 = st.sidebar.number_input(
    "Times 60-89 Days Late",
    min_value=0, max_value=20, value=0
)

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0, max_value=20, value=0
)

# ── Predict Button ────────────────────────────────────────
predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True)

# ── Main Panel ────────────────────────────────────────────
if predict_btn:

    # Build input dataframe
    input_data = pd.DataFrame([{
        'revolving_utilization': revolving_utilization,
        'age':                   age,
        'late_30_59':            late_30_59,
        'debt_ratio':            debt_ratio,
        'monthly_income':        monthly_income,
        'open_credit_lines':     open_credit_lines,
        'late_90_days':          late_90_days,
        'real_estate_loans':     real_estate_loans,
        'late_60_89':            late_60_89,
        'dependents':            dependents
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

    # Predict
    prob    = model.predict_proba(input_scaled_df)[0][1]
    pred    = model.predict(input_scaled_df)[0]
    risk_pct = prob * 100

    # ── Result ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Default Probability", f"{risk_pct:.1f}%")

    with col2:
        if risk_pct < 20:
            st.success("✅ LOW RISK")
        elif risk_pct < 50:
            st.warning("⚠️ MEDIUM RISK")
        else:
            st.error("🚨 HIGH RISK")

    with col3:
        st.metric("Decision", "Deny Loan" if pred == 1 else "Approve Loan")

    st.divider()

    # ── Risk Gauge ────────────────────────────────────────
    st.subheader("📊 Risk Assessment")

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(
        ['Risk'],
        [risk_pct],
        color='tomato' if risk_pct > 50 else 'orange' if risk_pct > 20 else 'steelblue',
        height=0.5
    )
    ax.barh(['Risk'], [100], color='lightgray', height=0.5, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Default Probability (%)')
    ax.axvline(x=20, color='green',  linestyle='--', alpha=0.5, label='Low/Med boundary')
    ax.axvline(x=50, color='red',    linestyle='--', alpha=0.5, label='Med/High boundary')
    ax.legend(loc='upper right', fontsize=8)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── SHAP Explanation ──────────────────────────────────
    st.subheader("🔍 Why This Prediction?")
    st.caption("SHAP values show how each feature pushed the prediction up or down")

    shap_single = explainer(input_scaled_df)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_single[0], show=False, max_display=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.divider()

    # ── Input Summary Table ───────────────────────────────
    st.subheader("📋 Applicant Summary")
    summary = input_data.T.reset_index()
    summary.columns = ['Feature', 'Value']
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    # Default landing page
    st.info("👈 Fill in the applicant details in the sidebar and click **Predict Risk**")

    st.subheader("ℹ️ About This App")
    st.markdown("""
    This app uses an **XGBoost** model trained on 150,000 loan applications 
    to predict the probability of a borrower defaulting within 2 years.

    **Model Performance:**
    - ROC-AUC Score: 0.85
    - Default Recall: 77%
    - Training samples: ~120,000

    **Features used:**
    - Revolving credit utilization
    - Age
    - Late payment history
    - Debt ratio
    - Monthly income
    - Credit lines and loans
    - Number of dependents
    """)