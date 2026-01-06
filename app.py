# =====================================
# STREAMLIT APP - TELCO CHURN PREDICTION
# =====================================

import streamlit as st
import pandas as pd
import joblib

# -------------------------------------
# LOAD MODEL
# -------------------------------------
# Model ini sudah berisi:
# - preprocessing (scaling + encoding)
# - model Random Forest terbaik (hasil GridSearch)
model = joblib.load("best_churn_model.pkl")

# -------------------------------------
# JUDUL APLIKASI
# -------------------------------------
st.title("📊 Prediksi Churn Pelanggan Telco")
st.write("""
Aplikasi ini digunakan untuk memprediksi apakah pelanggan **akan churn (berhenti)**  
atau **tetap berlangganan**, berdasarkan data layanan dan karakteristik pelanggan.
""")

st.markdown("---")

# -------------------------------------
# INPUT DATA PENGGUNA
# -------------------------------------
st.subheader("🧾 Masukkan Data Pelanggan")

tenure = st.number_input(
    "Lama Berlangganan (bulan)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Biaya Bulanan (Monthly Charges)",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Biaya (Total Charges)",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

gender = st.selectbox(
    "Jenis Kelamin",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Status Lansia",
    [0, 1],
    format_func=lambda x: "Ya" if x == 1 else "Tidak"
)

partner = st.selectbox(
    "Memiliki Pasangan?",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Memiliki Tanggungan?",
    ["Yes", "No"]
)

internet = st.selectbox(
    "Layanan Internet",
    ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Jenis Kontrak",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Metode Pembayaran",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

st.markdown("---")

# -------------------------------------
# DATAFRAME INPUT (HARUS SAMA DENGAN TRAINING)
# -------------------------------------
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "Yes",  # default aman
    "MultipleLines": "No",
    "InternetService": internet,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# -------------------------------------
# PREDIKSI
# -------------------------------------
if st.button("🔍 Prediksi Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"""❌ **PELANGGAN BERPOTENSI CHURN**
Probabilitas Churn: **{probability:.2%}**""")
    else:
        st.success(f"""✅ **PELANGGAN TIDAK CHURN**
Probabilitas Churn: **{probability:.2%}**""")
