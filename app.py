import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("best_churn_model.pkl")

# =====================================
# HEADER
# =====================================
st.markdown(
    """
    <h1 style='text-align: center;'>📊 Telco Customer Churn Prediction</h1>
    <p style='text-align: center; font-size: 18px;'>
    Prediksi apakah pelanggan akan <b>berhenti (Churn)</b> atau <b>tetap berlangganan</b>
    menggunakan Machine Learning.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Pengaturan Pelanggan")

gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Ya" if x == 1 else "Tidak"
)
partner = st.sidebar.selectbox("Memiliki Pasangan", ["Yes", "No"])
dependents = st.sidebar.selectbox("Memiliki Tanggungan", ["Yes", "No"])
internet = st.sidebar.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Metode Pembayaran",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# =====================================
# MAIN INPUT AREA
# =====================================
st.subheader("🧾 Informasi Tagihan")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Lama Berlangganan (bulan)", 0, 72, 12)

with col2:
    monthly_charges = st.number_input("Biaya Bulanan ($)", 0.0, 200.0, 70.0)

with col3:
    total_charges = st.number_input("Total Biaya ($)", 0.0, 10000.0, 1000.0)

st.markdown("---")

# =====================================
# DATAFRAME INPUT
# =====================================
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "Yes",
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

# =====================================
# PREDICTION
# =====================================
st.markdown("### 🔍 Hasil Prediksi")

if st.button("🚀 Prediksi Sekarang", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(
            f"""
            ❌ **PELANGGAN BERPOTENSI CHURN**  
            📉 Probabilitas Churn: **{probability:.2%}**
            """
        )
    else:
        st.success(
            f"""
            ✅ **PELANGGAN TIDAK CHURN**  
            📈 Probabilitas Churn: **{probability:.2%}**
            """
        )

# =====================================
# FOOTER
# =====================================
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px; color: gray;'>
    Developed by Awaludin | Telco Churn Prediction App | Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
