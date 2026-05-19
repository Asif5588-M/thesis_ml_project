# 🏥 High Cost Utilizer Prediction System
### Predicting High-Cost Lab Test Utilizers in a University Hospital Setting

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-App-yellow?logo=gradio)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **MS Health Informatics Thesis Project**
> A machine learning system to identify patients who are likely to become high-cost lab test utilizers, enabling proactive resource allocation and healthcare cost management.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Webapp](#-webapp-gradio)
- [Installation](#-installation)
- [Usage](#-usage)
- [Graphs & Visualizations](#-graphs--visualizations)
- [Future Work](#-future-work)

---

## 🔍 Overview

Healthcare systems worldwide face the challenge of **moral hazard** — where insured patients over-utilize diagnostic services. This project builds a predictive ML pipeline to flag patients at risk of becoming **High Cost Utilizers (HCU)** based on their lab test history.

**Key Question:** Can we predict, from a patient's visit and test patterns, whether they will fall into the top 20% of healthcare spenders?

**Three models were trained and compared:**
| Model | ROC-AUC | Accuracy | F1 Score |
|---|---|---|---|
| 🌲 Random Forest | **0.9901** | **94.82%** | **0.8711** |
| 🔵 Logistic Regression (Elastic Net) | 0.9826 | 92.07% | 0.8462 |
| 🔴 SVM (RBF Kernel) | 0.9670 | 91.38% | 0.8200 |

**Winner: Random Forest** with ROC-AUC of 0.9901.

---

## 📦 Dataset

- **Source:** University Hospital Lab records — `2024.xlsx`
- **Patients:** 2,380 unique patients
- **Period:** Year 2024 (monthly data)
- **Raw features:** Patient Name, Gender, Date, Tests ordered, Amount billed
- **Target:** `High_Cost_Utilizer` — top 20% by total amount (binary: 0/1)

**Class Distribution:**
```
Normal (0)   : 80%  → 1,904 patients
High Cost (1): 20%  →   476 patients
```
Class imbalance was handled using **SMOTE** on the training set only.

---

## 🔧 Feature Engineering

Raw transactional data was transformed into a patient-level ML-ready dataset with 12 features:

| Feature | Description |
|---|---|
| `Gender` | Male=1, Female=0 |
| `Visit_Count` | Number of unique visit dates |
| `Test_Freq` | Total number of tests ordered |
| `Test_Diversity` | Number of unique test types |
| `Repeat_Test_Count` | Test_Freq − Test_Diversity (repeated tests) |
| `num_BLOOD_tests` | Count of blood-category tests |
| `num_RADIOLOGY_tests` | Count of radiology tests (X-Ray, CT, MRI, U/S, etc.) |
| `num_URINE_tests` | Count of urine tests |
| `num_STOOL_tests` | Count of stool tests |
| `num_SPUTUM_tests` | Count of sputum/AFB tests |
| `Monthly_Test_Trend` | Increasing / Stable / Decreasing (encoded 2/1/0) |
| `Moral_Hazard_Index` | Composite risk score 0–3 (based on repeat, amount, frequency quantiles) |

**Top Predictors (Pearson correlation with target):**
```
Moral_Hazard_Index    : 0.756  ★ strongest
num_RADIOLOGY_tests   : 0.530
Test_Freq             : 0.507
Test_Diversity        : 0.503
Visit_Count           : 0.464
```

---

## 📁 Project Structure

```
thesis_ml_project/
│
├── data/
│   ├── 2024.xlsx                  # Raw hospital data
│   ├── final_ml_ready_health_data.csv  # Engineered features
│   ├── processed.csv              # Scaled (RobustScaler) data
│   └── processed_unscaled.csv     # Unscaled encoded data
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis + Feature Engineering
│   └── 02_models.ipynb            # Model Training, Evaluation & Comparison
│
├── models/
│   ├── lr_model.pkl               # Logistic Regression (best params)
│   ├── rf_model.pkl               # Random Forest (best params)
│   ├── svm_model.pkl              # SVM RBF (best params)
│   └── scaler.pkl                 # RobustScaler (fitted)
│
├── results/
│   ├── scaling_comparison.png     # Before/After RobustScaler histograms
│   ├── smote_comparison.png       # Class distribution before/after SMOTE
│   ├── lr_results.png             # LR Confusion Matrix + ROC Curve
│   ├── rf_results.png             # RF Confusion Matrix + ROC + Feature Importance
│   ├── svm_results.png            # SVM Confusion Matrix + ROC Curve
│   ├── model_comparison.png       # All 3 models — all metrics bar chart
│   ├── phase3_comparison.png      # Combined ROC + metrics + CM summary
│   ├── conf_matrix.png            # Individual confusion matrices (all 3)
│   ├── feature_importance.png     # RF feature importance (horizontal bar)
│   └── metrics.csv                # Final metrics table
│
└── webapp/
    ├── app.py                     # Gradio web application
    └── requirements.txt           # Python dependencies
```

---

## 🔬 Methodology

### Phase 1 — EDA & Feature Engineering (`01_EDA.ipynb`)

1. **Data Load & Inspection** — shape, dtypes, missing values, duplicates
2. **Feature Engineering** — patient-level aggregation from raw rows
3. **Target Variable** — `High_Cost_Utilizer` (top 20% by total amount = Rs. threshold)
4. **Moral Hazard Index** — composite 0–3 score from repeat tests, total amount, and test frequency quantiles
5. **Correlation Analysis** — Pearson correlations, heatmap
6. **Outlier Detection** — IQR method; decision: **keep outliers** (genuine high-cost patients; RobustScaler handles them)
7. **Preprocessing** — Label encoding for `Monthly_Test_Trend`, **RobustScaler** applied
8. **Save** — `processed.csv` and `processed_unscaled.csv`

### Phase 2 — Model Training (`02_models.ipynb`)

1. **Train/Test Split** — 70/30 stratified split
2. **SMOTE** — applied on training set only to balance classes (80/20 → 50/50)
3. **GridSearchCV** — 5-fold CV, scoring = `roc_auc`

| Model | Hyperparameters Tuned |
|---|---|
| Logistic Regression | `C` ∈ {0.01, 0.1, 1, 10}, `l1_ratio` ∈ {0.1–0.9} |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| SVM (RBF) | `C` ∈ {0.1, 1, 10, 100}, `gamma` ∈ {scale, auto, 0.01, 0.1} |

4. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC on original (unbalanced) test set
5. **Model Saving** — `joblib` → `.pkl` files

### Phase 3 — Comparison

- Combined ROC curves
- Individual and summary confusion matrices
- Feature importance (Random Forest)
- Final verdict: **Random Forest best overall; Logistic Regression best Recall (0.9371)**

---

## 📊 Results

### Final Model Comparison

```
                    Accuracy  Precision  Recall    F1      ROC_AUC
Logistic Regression  0.9207    0.7982    0.9371  0.8621    0.9826
Random Forest        0.9482    0.8681    0.8739  0.8710    0.9901  ← Best
SVM                  0.9138    0.7900    0.8571  0.8222    0.9670
```

### Clinical Interpretation

- **Random Forest** — best for overall triage (highest AUC + precision)
- **Logistic Regression** — best when **missing a high-cost patient is costly** (highest Recall = 93.7% of true HCU patients caught)
- **SVM** — solid but outperformed by both alternatives

---

## 🌐 Webapp (Gradio)

An interactive prediction webapp was built using **Gradio** that allows clinicians to enter patient data and get real-time predictions from any of the 3 models.

**Features:**
- Select model (RF / LR / SVM)
- Adjust all 12 patient features via sliders/dropdowns
- Instant prediction with confidence % and probability breakdown
- Patient summary table

**Run locally:**
```bash
cd webapp
python app.py
# Opens at http://localhost:7860
```

---

## 💻 Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/high-cost-utilizer-prediction.git
cd high-cost-utilizer-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r webapp/requirements.txt
```

**Dependencies:**
```
gradio
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
joblib
openpyxl
```

---

## 🚀 Usage

### Run the Notebooks (in order)

```bash
jupyter notebook
# Open: notebooks/01_EDA.ipynb   → Run all cells
# Open: notebooks/02_models.ipynb → Run all cells
```

> ⚠️ Make sure `data/2024.xlsx` is present before running `01_EDA.ipynb`

### Run the Webapp

```bash
cd webapp
python app.py
```

### Load a saved model in Python

```python
import joblib
import numpy as np

# Load model + scaler
rf_model = joblib.load('models/rf_model.pkl')
scaler   = joblib.load('models/scaler.pkl')

# Example patient (all 12 features)
patient = np.array([[1, 5, 18, 12, 6, 14, 2, 1, 0, 0, 1, 2]])
patient_scaled = scaler.transform(patient)

pred = rf_model.predict(patient_scaled)
prob = rf_model.predict_proba(patient_scaled)

print("Prediction:", "High Cost" if pred[0]==1 else "Normal")
print(f"Probability: {prob[0][1]*100:.1f}%")
```

---

## 📈 Graphs & Visualizations

All result graphs are saved in the `results/` folder:

| File | Description |
|---|---|
| `scaling_comparison.png` | Histograms before/after RobustScaler for all 12 features |
| `smote_comparison.png` | Class distribution: original → train before SMOTE → train after SMOTE |
| `lr_results.png` | LR confusion matrix + ROC curve |
| `rf_results.png` | RF confusion matrix + ROC curve + feature importance |
| `svm_results.png` | SVM confusion matrix + ROC curve |
| `model_comparison.png` | Grouped bar chart — all 5 metrics × 3 models |
| `conf_matrix.png` | Side-by-side confusion matrices (Blues/Greens/Reds) |
| `feature_importance.png` | RF feature importances (top features highlighted in red) |

> **Graph Adjustment Notes** *(for research paper):*
> - `figsize` can be increased for publication quality (e.g. `(18,8)` → `(20,10)`)
> - `dpi=150` → `dpi=300` for print-ready output
> - Fonts can be scaled: `plt.rcParams['font.size'] = 14`
> - Color palettes can be changed: Blues/Greens/Reds → any matplotlib colormap

---

## 🔮 Future Work

- [ ] Include patient age/diagnosis data for richer features
- [ ] Test XGBoost / LightGBM as additional models
- [ ] Longitudinal analysis across multiple years
- [ ] Deploy webapp on HuggingFace Spaces
- [ ] SHAP values for model explainability
- [ ] Threshold tuning for clinical recall targets

---

## 👩‍🔬 Research Context

This project was developed as part of an **MS Health Informatics thesis** investigating the application of machine learning to healthcare cost prediction in a Pakistani university hospital setting.

The **Moral Hazard Index (MHI)** is a novel composite feature developed for this study, combining three behavioral signals — repeat test frequency, total spending, and test ordering volume — into a single 0–3 risk score that proved to be the **strongest single predictor** (r = 0.756) of high-cost utilization.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Made with ❤️ for MS Health Informatics Thesis*