# 🏥 Healthcare High-Cost Utilizer Prediction System

> ML-based Early Warning System for University Medical Centers

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)]()
[![Published](https://img.shields.io/badge/Paper-HEC%20Y--Category-green)]()

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🚀 Live Demo | [asif-healthcare-predictor.streamlit.app](https://asif-healthcare-predictor.streamlit.app) |
| 📄 Published Paper | [HEC Y-Category Journal](https://amresearchjournal.com/index.php/Journal/article/view/1847) |
| 💼 LinkedIn | [Asif Nawaz](https://www.linkedin.com/in/asif-nawaz-75042a147) |

---

## 📌 Problem Statement

University medical centers in Pakistan have no system
to predict which patients will become high-cost
utilizers before treatment begins.

At Arid Agriculture University Rawalpindi Medical
Center, this caused:
- Budget overruns
- Diagnostic duplication
- No early warning system
- Economic waste in lab testing

---

## ✅ Solution

Machine Learning models that predict — using patient
lab test history — whether a patient will become a
**High-Cost Utilizer before treatment**.

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Random Forest** | 94.82% | 86.81% | 87.41% | 87.11% | **0.9901 🏆** |
| Logistic Regression | 94.96% | 83.23% | 93.71% | 88.16% | 0.9826 |
| SVM | 93.98% | 82.89% | 88.11% | 85.42% | 0.9669 |

**Best Model: Random Forest — ROC-AUC 0.9901**

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| Visit Count | Total hospital visits |
| Test Frequency | Total lab tests ordered |
| Test Diversity | Unique test types |
| Repeat Test Count | Duplicate tests |
| Moral Hazard Index | Risk of overutilization |
| Monthly Test Trend | Increasing/Stable/Decreasing |

---

## 🧠 ML Pipeline
Raw Data → EDA → Feature Engineering
→ Train/Test Split → SMOTE
→ GridSearchCV → Model Training
→ Evaluation → Streamlit Deployment

---

## 🛠️ Tech Stack
Python 3.12
Scikit-learn    — ML Models
Imbalanced-learn — SMOTE
Pandas, NumPy   — Data Processing
Matplotlib      — Visualization
Streamlit       — Web App
Joblib          — Model Saving

---

## 📁 Project Structure
thesis_ml_project/
├── data/
│   ├── processed.csv
│   └── final_ml_ready_health_data.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_models.ipynb
├── models/
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl
├── webapp/
│   └── streamlit_app.py
├── results/
│   ├── lr_results.png
│   ├── rf_results.png
│   └── svm_results.png
└── requirements.txt

---

## 🚀 How to Run

```bash
git clone https://github.com/Asif5588-M/thesis_ml_project
cd thesis_ml_project
pip install -r requirements.txt
cd webapp
streamlit run streamlit_app.py
```

---

## 📄 Published Research

**"Machine Learning for Sustainable Healthcare:
Identifying High-Cost Utilizers and Diagnostic
Waste in Pakistan"**

— HEC Y-Category Journal

🔗 [Read Paper](https://amresearchjournal.com/index.php/Journal/article/view/1847)

---

## 👨‍💻 Author

**Asif Nawaz**

🏥 Healthcare Data Scientist
📚 MPhil Economics — PMAS Arid Agriculture University
🔬 13+ Years Clinical Experience
📄 Published Researcher — HEC Y-Category

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/asif-nawaz-75042a147)