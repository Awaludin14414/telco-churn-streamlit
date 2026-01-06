import streamlit as st
import pandas as pd
import joblib
import time

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("best_churn_model.pkl")

# =====================================
# SESSION STATE (RIWAYAT)
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# HEADER
# =====================================
st.markdown("""
<h1 style='text-align:center;'>📊 Telco Customer Churn Dashboard</h1>
<p style='text-align:center; font-size:18px;'>
Aplikasi prediksi churn pelanggan menggunakan <b>Machine Learning</b>
</p>
<hr>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR INPUT
# =====================================
st.sidebar.header("⚙️ Data Pelanggan")

gender = st.sidebar.selectbox("Jenis Kelamin", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
partner = st.sidebar.selectbox("Memiliki Pasangan", ["Yes", "No"])
dependents = st.sidebar.selectbox("Memiliki Tanggungan", ["Yes", "No"])
internet = st.sidebar.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Metode Pembayaran",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# =====================================
# MAIN INPUT AREA
# =====================================
st.subheader("🧾 Informasi Tagihan")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input("Tenure (bulan)", 0, 72, 12)
with col2:
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
with col3:
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# =====================================
# DATAFRAME INPUT
# =====================================
input_df = pd.DataFrame([{
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
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

st.markdown("---")

# =====================================
# PREDIKSI
# =====================================
if st.button("🚀 Prediksi Sekarang", use_container_width=True):
    with st.spinner("⏳ Memproses prediksi..."):
        time.sleep(1)
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

    # PROGRESS BAR
    st.subheader("📈 Probabilitas Churn")
    st.progress(int(prob * 100))
    st.write(f"**Probabilitas Churn: {prob:.2%}**")

    # VISUAL GAUGE
    if prob < 0.3:
        gauge = "🟢🟢🟢🟢🟢"
        risk = "RISIKO RENDAH"
    elif prob < 0.6:
        gauge = "🟡🟡🟡🟡"
        risk = "RISIKO SEDANG"
    else:
        gauge = "🔴🔴🔴🔴🔴"
        risk = "RISIKO TINGGI"

    st.markdown(f"### {gauge}")
    st.info(f"**Tingkat Risiko: {risk}**")

    # HASIL AKHIR
    st.markdown("---")
    if pred == 1:
        st.error("❌ **PELANGGAN DIPREDIKSI CHURN**")
    else:
        st.success("✅ **PELANGGAN DIPREDIKSI TIDAK CHURN**")

    # SIMPAN RIWAYAT
    st.session_state.history.append({
        "Tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Probabilitas Churn": f"{prob:.2%}",
        "Hasil": "Churn" if pred == 1 else "Tidak Churn"
    })

# =====================================
# RIWAYAT PREDIKSI
# =====================================
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Riwayat Prediksi (Session)")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px; color:gray;'>
Developed by <b>Awaludin</b> | Telco Churn Prediction Dashboard | Streamlit
</p>
""", unsafe_allow_html=True)
