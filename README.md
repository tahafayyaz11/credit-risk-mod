# 🏦 Credit Risk Predictor

An end-to-end machine learning project that predicts whether a loan applicant 
will default on their loan within 2 years.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red)

---

## 🔗 Live Demo
[Try the app here](#) ← replace with Hugging Face link after deployment

---

## 📊 Model Performance

| Model | ROC-AUC | Default Recall |
|-------|---------|----------------|
| Logistic Regression | 0.79 | 0.71 |
| Random Forest | 0.82 | 0.11 |
| **XGBoost** | **0.85** | **0.77** |

XGBoost was selected as the final model due to best overall performance.

---

## 🗂️ Project Structure
credit-risk-model/
│
├── app/
│   └── app.py              ← Streamlit web application
│
├── data/
│   └── processed/          ← cleaned and scaled data
│
├── notebooks/
│   ├── 01_eda.ipynb         ← exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb    ← model training and comparison
│   └── 04_shap.ipynb        ← explainability analysis
│
├── models/                  ← saved model and scaler
├── requirements.txt
└── README.md
---

## 📁 Dataset

- **Source:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Size:** 150,000 loan applications
- **Target:** `SeriousDlqin2yrs` — whether borrower defaulted within 2 years
- **Class balance:** 93.3% no default / 6.7% default

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Web App | Streamlit |
| Deployment | Hugging Face Spaces |

---

## 🔍 Key Features

- **Full ML pipeline** — EDA → Preprocessing → Modeling → Deployment
- **3 models compared** — Logistic Regression, Random Forest, XGBoost
- **Class imbalance handling** — using `scale_pos_weight` in XGBoost
- **SHAP explainability** — understand WHY each prediction was made
- **Live web app** — interactive UI to test any applicant

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/tahafayyaz11/credit-risk-model.git
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## 📈 What I Learned

- Handling severe class imbalance in real-world data
- Why accuracy is a misleading metric for imbalanced datasets
- How XGBoost outperforms traditional models on tabular data
- Using SHAP values to make ML models explainable
- End-to-end deployment of ML models with Streamlit

---

## 👤 Author

**Taha Fayyaz**  
[GitHub](https://github.com/tahafayyaz11)
