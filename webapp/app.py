
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lr_model  = joblib.load(os.path.join(BASE_DIR, "models", "lr_model.pkl"))
rf_model  = joblib.load(os.path.join(BASE_DIR, "models", "rf_model.pkl"))
svm_model = joblib.load(os.path.join(BASE_DIR, "models", "svm_model.pkl"))
scaler    = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

print("All models loaded successfully!")

# ── Prediction function ───────────────────────────────────────
def predict(gender, visit_count, test_freq, test_diversity,
            repeat_test_count, num_blood, num_radiology,
            num_urine, num_stool, num_sputum,
            monthly_trend, moral_hazard, model_choice):

    trend_map     = {"Decreasing": 0, "Stable": 1, "Increasing": 2}
    trend_encoded = trend_map[monthly_trend]

    features = np.array([[
        gender, visit_count, test_freq, test_diversity,
        repeat_test_count, num_blood, num_radiology,
        num_urine, num_stool, num_sputum,
        trend_encoded, moral_hazard
    ]])

    features_scaled = scaler.transform(features)

    if model_choice == "Logistic Regression":
        model = lr_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = svm_model

    pred  = model.predict(features_scaled)[0]
    prob  = model.predict_proba(features_scaled)[0]

    label      = "HIGH COST UTILIZER" if pred == 1 else "NORMAL UTILIZER"
    confidence = f"{max(prob)*100:.1f}%"
    status     = "WARNING" if pred == 1 else "OK"

    result = f"""
## Prediction Result — {status}

**{label}**
**Confidence:** {confidence}

---
### Probability Breakdown
- Normal (0)    : {prob[0]*100:.1f}%
- High Cost (1) : {prob[1]*100:.1f}%

---
### Patient Summary
| Feature              | Value                            |
|----------------------|----------------------------------|
| Gender               | {"Male" if gender == 1 else "Female"} |
| Visit Count          | {visit_count}                    |
| Test Frequency       | {test_freq}                      |
| Test Diversity       | {test_diversity}                 |
| Repeat Test Count    | {repeat_test_count}              |
| Blood Tests          | {num_blood}                      |
| Radiology Tests      | {num_radiology}                  |
| Urine Tests          | {num_urine}                      |
| Stool Tests          | {num_stool}                      |
| Sputum Tests         | {num_sputum}                     |
| Monthly Trend        | {monthly_trend}                  |
| Moral Hazard Index   | {moral_hazard}                   |

---
*Model used: {model_choice}*
*Best model: Random Forest (ROC-AUC = 0.990)*
"""
    return result

# ── Gradio Interface ──────────────────────────────────────────
with gr.Blocks(title="High Cost Utilizer Predictor",
               theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # High Cost Utilizer Prediction System
    ### Predicting High-Cost Lab Test Utilizers — University Hospital
    **Thesis Project | MS Health Informatics**
    """)

    with gr.Row():

        # Left column — inputs
        with gr.Column(scale=1):

            gr.Markdown("### Patient Info")
            gender = gr.Radio(
                choices=[["Male", 1], ["Female", 0]],
                label="Gender", value=1)
            model_choice = gr.Dropdown(
                choices=["Random Forest", "Logistic Regression", "SVM"],
                label="Select Model", value="Random Forest")

            gr.Markdown("### Visit & Test Info")
            visit_count = gr.Slider(
                1, 13, value=1, step=1, label="Visit Count")
            test_freq = gr.Slider(
                1, 60, value=4, step=1, label="Test Frequency (total tests)")
            test_diversity = gr.Slider(
                1, 25, value=4, step=1, label="Test Diversity (unique tests)")
            repeat_test_count = gr.Slider(
                0, 31, value=0, step=1, label="Repeat Test Count")

            gr.Markdown("### Test Categories")
            num_blood = gr.Slider(
                0, 46, value=3, step=1, label="Blood Tests")
            num_radiology = gr.Slider(
                0, 5, value=0, step=1, label="Radiology Tests")
            num_urine = gr.Slider(
                0, 6, value=0, step=1, label="Urine Tests")
            num_stool = gr.Slider(
                0, 2, value=0, step=1, label="Stool Tests")
            num_sputum = gr.Slider(
                0, 2, value=0, step=1, label="Sputum Tests")

            gr.Markdown("### Trend & Risk")
            monthly_trend = gr.Dropdown(
                choices=["Stable", "Increasing", "Decreasing"],
                label="Monthly Test Trend", value="Stable")
            moral_hazard = gr.Slider(
                1, 3, value=1, step=1,
                label="Moral Hazard Index (1=Low, 2=Medium, 3=High)")

            predict_btn = gr.Button(
                "Predict", variant="primary", size="lg")

        # Right column — output
        with gr.Column(scale=1):
            gr.Markdown("### Prediction Output")
            output = gr.Markdown(
                value="*Fill in patient details and click Predict*")

    predict_btn.click(
        fn=predict,
        inputs=[gender, visit_count, test_freq, test_diversity,
                repeat_test_count, num_blood, num_radiology,
                num_urine, num_stool, num_sputum,
                monthly_trend, moral_hazard, model_choice],
        outputs=output
    )

    gr.Markdown("""
    ---
    **Models:** Random Forest (Best — AUC 0.990) |
    Logistic Regression (AUC 0.983) | SVM (AUC 0.967)
    """)

if __name__ == "__main__":
    app.launch(share=False)
