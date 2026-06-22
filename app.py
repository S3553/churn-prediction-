
import streamlit as st
import numpy as np
import joblib

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title-box {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white;
    text-align:center;
    margin-bottom:20px;
}

.metric-card{
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
}

.result-box{
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown("""
<div class='title-box'>
<h1>📊 Customer Churn Prediction Dashboard</h1>
<p>Predict whether a customer is likely to leave the company.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# TOP CARDS
# -----------------------
c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class='metric-card'>
    <h3>21</h3>
    <p>Features</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='metric-card'>
    <h3>SVM</h3>
    <p>Model</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='metric-card'>
    <h3>Churn</h3>
    <p>Target</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class='metric-card'>
    <h3>ML</h3>
    <p>Prediction</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------
# INPUT SECTION
# -----------------------

left,right = st.columns(2)

with left:

    st.subheader("👤 Customer Information")

    SeniorCitizen = st.selectbox("Senior Citizen",["No","Yes"])
    Partner = st.selectbox("Partner",["No","Yes"])
    Dependents = st.selectbox("Dependents",["No","Yes"])

    tenure = st.slider("Tenure (Months)",0,72,12)

    MultipleLines = st.selectbox("Multiple Lines",["No","Yes"])
    OnlineSecurity = st.selectbox("Online Security",["No","Yes"])
    OnlineBackup = st.selectbox("Online Backup",["No","Yes"])

    DeviceProtection = st.selectbox("Device Protection",["No","Yes"])
    TechSupport = st.selectbox("Tech Support",["No","Yes"])

with right:

    st.subheader("📺 Services & Billing")

    StreamingTV = st.selectbox("Streaming TV",["No","Yes"])
    StreamingMovies = st.selectbox("Streaming Movies",["No","Yes"])

    PaperlessBilling = st.selectbox("Paperless Billing",["No","Yes"])

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1500.0
    )

    InternetService_Fiber_optic = st.selectbox(
        "Fiber Optic Internet",
        ["No","Yes"]
    )

    InternetService_No = st.selectbox(
        "No Internet Service",
        ["No","Yes"]
    )

    Contract_One_year = st.selectbox(
        "One Year Contract",
        ["No","Yes"]
    )

    Contract_Two_year = st.selectbox(
        "Two Year Contract",
        ["No","Yes"]
    )

    Payment_Credit = st.selectbox(
        "Credit Card Payment",
        ["No","Yes"]
    )

    Payment_Electronic = st.selectbox(
        "Electronic Check",
        ["No","Yes"]
    )

    Payment_Mailed = st.selectbox(
        "Mailed Check",
        ["No","Yes"]
    )

# -----------------------
# CONVERT YES/NO
# -----------------------

def yn(x):
    return 1 if x=="Yes" else 0

# -----------------------
# PREDICT BUTTON
# -----------------------

st.write("")
predict = st.button(
    "🚀 Predict Churn",
    use_container_width=True
)

if predict:

    features = np.array([[
        yn(SeniorCitizen),
        yn(Partner),
        yn(Dependents),
        tenure,
        yn(MultipleLines),
        yn(OnlineSecurity),
        yn(OnlineBackup),
        yn(DeviceProtection),
        yn(TechSupport),
        yn(StreamingTV),
        yn(StreamingMovies),
        yn(PaperlessBilling),
        MonthlyCharges,
        TotalCharges,
        yn(InternetService_Fiber_optic),
        yn(InternetService_No),
        yn(Contract_One_year),
        yn(Contract_Two_year),
        yn(Payment_Credit),
        yn(Payment_Electronic),
        yn(Payment_Mailed)
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    risk = probability * 100

    st.write("")
    st.subheader("📈 Prediction Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class='result-box'
            style='background:#7f1d1d;color:white'>
            ⚠️ HIGH RISK CUSTOMER
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class='result-box'
            style='background:#14532d;color:white'>
            ✅ CUSTOMER LIKELY TO STAY
            </div>
            """,
            unsafe_allow_html=True
        )

    st.metric(
        "Churn Probability",
        f"{risk:.2f}%"
    )

    st.progress(min(int(risk),100))

# -----------------------
# FOOTER
# -----------------------

st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit, Scikit-Learn and Support Vector Machine"
)
