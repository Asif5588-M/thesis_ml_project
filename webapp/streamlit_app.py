import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="High Cost Utilizer Prediction",
    page_icon="🏥",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    lr  = joblib.load(os.path.join(BASE_DIR, "models", "lr_model.pkl"))
    rf  = joblib.load(os.path.join(BASE_DIR, "models", "rf_model.pkl"))
    svm = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    return lr, rf, svm, scaler

lr_model, rf_model, svm_model, scaler = load_models()

# ── Profile Header ────────────────────────────────────────────
col_img, col_info = st.columns([1, 8])

with col_img:
    profile_path = os.path.join(WEBAPP_DIR, "profile.png")
    if os.path.exists(profile_path):
        st.image(profile_path, width=80)

with col_info:
    st.markdown("### Asif Nawaz")
    st.markdown(
        "🏥 Healthcare Data Scientist &nbsp;|&nbsp; "
        "MPhil Economics &nbsp;|&nbsp; "
        "Published Researcher &nbsp;|&nbsp; "
        "13+ Years Clinical Experience"
    )

st.divider()

# ── Main Title ────────────────────────────────────────────────
st.title("🏥 High Cost Utilizer Prediction System")
st.markdown(
    "**Predicting High-Cost Lab Test Utilizers — "
    "Arid Agriculture University Rawalpindi**"
)
st.divider()

# ── Layout ────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Information")

    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_val = 1 if gender == "Male" else 0

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "SVM"])

    st.subheader("Visit & Test Info")

    visit_count = st.slider("Visit Count", 1, 13, 1)
    test_freq = st.slider("Test Frequency (total tests)", 1, 60, 4)
    test_diversity = st.slider("Test Diversity (unique tests)", 1, 25, 4)
    repeat_test_count = st.slider("Repeat Test Count", 0, 31, 0)

    st.subheader("Test Categories")

    num_blood = st.slider("Blood Tests", 0, 46, 3)
    num_radiology = st.slider("Radiology Tests", 0, 5, 0)
    num_urine = st.slider("Urine Tests", 0, 6, 0)
    num_stool = st.slider("Stool Tests", 0, 2, 0)
    num_sputum = st.slider("Sputum Tests", 0, 2, 0)

    st.subheader("Trend & Risk")

    monthly_trend = st.selectbox(
        "Monthly Test Trend",
        ["Stable", "Increasing", "Decreasing"])
    trend_map = {"Decreasing": 0, "Stable": 1, "Increasing": 2}
    trend_val = trend_map[monthly_trend]

    moral_hazard = st.slider(
        "Moral Hazard Index (1=Low, 2=Medium, 3=High)", 1, 3, 1)

    predict_btn = st.button(
        "🔍 Predict", 
        type="primary",
        use_container_width=True)

# ── Prediction Output ─────────────────────────────────────────
with col2:
    st.subheader("Prediction Output")

    if predict_btn:
        features = np.array([[
            gender_val, visit_count, test_freq, test_diversity,
            repeat_test_count, num_blood, num_radiology,
            num_urine, num_stool, num_sputum,
            trend_val, moral_hazard
        ]])

        features_scaled = scaler.transform(features)

        if model_choice == "Random Forest":
            model = rf_model
        elif model_choice == "Logistic Regression":
            model = lr_model
        else:
            model = svm_model

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        # ── Result Box ────────────────────────────────────────
        if pred == 1:
            st.error("⚠️ HIGH COST UTILIZER")
        else:
            st.success("✅ NORMAL UTILIZER")

        # ── Confidence ────────────────────────────────────────
        st.metric("Confidence", f"{max(prob)*100:.1f}%")

        # ── Probability ───────────────────────────────────────
        st.subheader("Probability Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Normal (0)", f"{prob[0]*100:.1f}%")
        with col_b:
            st.metric("High Cost (1)", f"{prob[1]*100:.1f}%")

        # ── Patient Summary ───────────────────────────────────
        st.subheader("Patient Summary")
        summary = {
            "Feature": [
                "Gender", "Visit Count", "Test Frequency",
                "Test Diversity", "Repeat Tests",
                "Blood Tests", "Radiology Tests",
                "Urine Tests", "Stool Tests", "Sputum Tests",
                "Monthly Trend", "Moral Hazard Index"
            ],
            "Value": [
                gender, visit_count, test_freq,
                test_diversity, repeat_test_count,
                num_blood, num_radiology,
                num_urine, num_stool, num_sputum,
                monthly_trend, moral_hazard
            ]
        }
        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True)

        st.caption(
            f"Model: {model_choice} | "
            f"Best Model: Random Forest (ROC-AUC = 0.990)")

    else:
        st.info("👈 Fill in patient details and click Predict")

        # ── Model Performance Table ───────────────────────────
        st.subheader("Model Performance")
        perf_data = {
            "Model": [
                "Random Forest", 
                "Logistic Regression", 
                "SVM"],
            "Accuracy": ["94.82%", "94.96%", "93.98%"],
            "Precision": ["86.81%", "83.23%", "82.89%"],
            "Recall": ["87.41%", "93.71%", "88.11%"],
            "ROC-AUC": ["0.9901 🏆", "0.9826", "0.9669"]
        }
        st.dataframe(
            pd.DataFrame(perf_data),
            use_container_width=True,
            hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "👨‍💻 Asif Nawaz | MPhil Economics | "
    "Arid Agriculture University Rawalpindi | "
    "📄 Published: HEC Y-Category Journal | "
    "🏥 13+ Years Healthcare Experience"
)